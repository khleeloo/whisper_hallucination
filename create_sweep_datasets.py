"""
Create noisy datasets for a noise-ratio sweep experiment.

Generates (noise_type x noise_ratio) combinations using the same clean subset
as the main experiment for consistency.

Noisy rows SUBSTITUTE (replace) a fraction of clean rows — dataset stays the
same total size as the clean baseline. A 24% noise dataset with 120k clean →
91.2k clean + 28.8k noisy = 120k total.

Noise types: UU (unique-unique), RR (repeat-repeat), RU (repeat-unique), UR (unique-repeat)
Noise ratios: 1%, 2%, 5%, 10%, 20%, 50%

Usage:
    python create_sweep_datasets.py \
        --data_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en \
        --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
        --subset_size 120000 \
        --seed 42

    # Specific types/ratios with bare directory names:
    python create_sweep_datasets.py \\
        --data_dir ... \\
        --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_24pct \\
        --subset_size 120000 \\
        --noise_types rr ru \\
        --noise_ratios 0.24 \\
        --name_template bare \\
        --seed 42
"""

import argparse
import os
import random

from create_noisy_dataset import (
    load_tsv,
    write_tsv,
    create_subset,
    create_noise_uu,
    create_noise_rr,
    create_noise_ru,
    create_noise_ur,
)

NOISE_TYPES = ["uu", "rr", "ru", "ur"]
NOISE_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeat_pairs", type=int, default=10)
    parser.add_argument("--noise_types", type=str, nargs="+", default=None,
                        help=f"Noise types to generate (default: all = {NOISE_TYPES})")
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=None,
                        help=f"Noise ratios to generate (default: all = {NOISE_RATIOS})")
    parser.add_argument("--name_template", type=str, default="ratio",
                        choices=["ratio", "bare"],
                        help="ratio = names like rr_24 (default); bare = names like rr "
                             "(use when each noise_type has exactly one ratio)")
    args = parser.parse_args()

    noise_types = args.noise_types if args.noise_types is not None else NOISE_TYPES
    noise_ratios = args.noise_ratios if args.noise_ratios is not None else NOISE_RATIOS

    for nt in noise_types:
        if nt not in NOISE_TYPES:
            parser.error(f"Unknown noise type '{nt}'. Choose from {NOISE_TYPES}")

    # Load training data
    train_tsv = os.path.join(args.data_dir, "train.tsv")
    print(f"Loading {train_tsv}...")
    fieldnames, all_rows = load_tsv(train_tsv)
    print(f"  Total training utterances: {len(all_rows)}")

    # Create the SAME clean subset used in the main experiment (same seed)
    rng = random.Random(args.seed)
    clean_rows = create_subset(all_rows, args.subset_size, rng)
    print(f"  Clean subset size: {len(clean_rows)}")
    print(f"  Noise types: {noise_types}")
    print(f"  Noise ratios: {noise_ratios}")

    noise_fns = {
        "uu": create_noise_uu,
        "rr": create_noise_rr,
        "ru": create_noise_ru,
        "ur": create_noise_ur,
    }

    REPEAT_NOISE_TYPES = {"rr", "ru", "ur"}

    for noise_type in noise_types:
        for ratio in noise_ratios:
            pct = int(ratio * 100)
            if args.name_template == "bare":
                config_name = noise_type
            else:
                config_name = f"{noise_type}_{pct:02d}"
            n_noisy = int(len(clean_rows) * ratio)

            print(f"\n=== {config_name}: {noise_type.upper()} at {pct}% ({n_noisy} noisy samples) ===")

            # Fresh RNG per config for reproducibility
            config_rng = random.Random(args.seed + hash(config_name))

            if noise_type == "rr":
                noisy_rows = noise_fns[noise_type](
                    clean_rows, n_noisy, config_rng, n_pairs=args.n_repeat_pairs
                )
            elif noise_type == "ru":
                noisy_rows = noise_fns[noise_type](
                    clean_rows, n_noisy, config_rng, n_audios=args.n_repeat_pairs
                )
            elif noise_type == "ur":
                noisy_rows = noise_fns[noise_type](
                    clean_rows, n_noisy, config_rng, n_sentences=args.n_repeat_pairs
                )
            else:
                noisy_rows = noise_fns[noise_type](clean_rows, n_noisy, config_rng)

            # Substitute: replace a fraction of clean rows with noisy mismatches.
            # Keep (1 - ratio) clean + ratio noisy = same total size as baseline.
            n_keep = len(clean_rows) - n_noisy
            kept_clean = config_rng.sample(clean_rows, n_keep)
            combined = kept_clean + noisy_rows

            config_rng.shuffle(combined)

            config_dir = os.path.join(args.output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            write_tsv(os.path.join(config_dir, "train.tsv"), fieldnames, combined)
            write_tsv(os.path.join(config_dir, "noisy_only.tsv"), fieldnames, noisy_rows)

            print(f"  {n_keep} clean + {len(noisy_rows)} noisy = {len(combined)} total "
                  f"({pct}% noise, same size)")

    # Summary
    print("\n=== Sweep Summary ===")
    print(f"Clean subset: {len(clean_rows)} utterances")
    print(f"All datasets stay at {len(clean_rows)} total (substitution mode)")
    print(f"Configs created: {len(noise_types) * len(noise_ratios)}")
    for noise_type in noise_types:
        for ratio in noise_ratios:
            pct = int(ratio * 100)
            n_noisy = int(len(clean_rows) * ratio)
            print(f"  {noise_type}_{pct:02d}: {len(clean_rows)} total ({n_noisy} noisy)")
    print("Done!")


if __name__ == "__main__":
    main()
