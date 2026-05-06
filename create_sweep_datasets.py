"""
Create noisy datasets for a noise-ratio sweep experiment.

Generates (noise_type x noise_ratio) combinations using the same clean subset
as the main experiment for consistency.

Noise types: UU (unique-unique), RR (repeat-repeat)
Noise ratios: 1%, 2%, 5%, 10%, 20%, 50%

Usage:
    python create_sweep_datasets.py \
        --data_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en \
        --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
        --subset_size 120000 \
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
)

NOISE_TYPES = ["uu", "rr"]
NOISE_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeat_pairs", type=int, default=10)
    args = parser.parse_args()

    # Load training data
    train_tsv = os.path.join(args.data_dir, "train.tsv")
    print(f"Loading {train_tsv}...")
    fieldnames, all_rows = load_tsv(train_tsv)
    print(f"  Total training utterances: {len(all_rows)}")

    # Create the SAME clean subset used in the main experiment (same seed)
    rng = random.Random(args.seed)
    clean_rows = create_subset(all_rows, args.subset_size, rng)
    print(f"  Clean subset size: {len(clean_rows)}")

    noise_fns = {
        "uu": create_noise_uu,
        "rr": create_noise_rr,
    }

    for noise_type in NOISE_TYPES:
        for ratio in NOISE_RATIOS:
            pct = int(ratio * 100)
            config_name = f"{noise_type}_{pct:02d}"
            n_noisy = int(len(clean_rows) * ratio)

            print(f"\n=== {config_name}: {noise_type.upper()} at {pct}% ({n_noisy} noisy samples) ===")

            # Fresh RNG per config for reproducibility
            config_rng = random.Random(args.seed + hash(config_name))

            if noise_type == "rr":
                noisy_rows = noise_fns[noise_type](
                    clean_rows, n_noisy, config_rng, n_pairs=args.n_repeat_pairs
                )
            else:
                noisy_rows = noise_fns[noise_type](clean_rows, n_noisy, config_rng)

            combined = list(clean_rows) + noisy_rows
            config_rng.shuffle(combined)

            config_dir = os.path.join(args.output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            write_tsv(os.path.join(config_dir, "train.tsv"), fieldnames, combined)
            write_tsv(os.path.join(config_dir, "noisy_only.tsv"), fieldnames, noisy_rows)

            print(f"  {len(clean_rows)} clean + {len(noisy_rows)} noisy = {len(combined)} total")

    # Summary
    print("\n=== Sweep Summary ===")
    print(f"Clean subset: {len(clean_rows)} utterances")
    print(f"Configs created: {len(NOISE_TYPES) * len(NOISE_RATIOS)}")
    for noise_type in NOISE_TYPES:
        for ratio in NOISE_RATIOS:
            pct = int(ratio * 100)
            n_noisy = int(len(clean_rows) * ratio)
            print(f"  {noise_type}_{pct:02d}: {len(clean_rows) + n_noisy} total ({n_noisy} noisy)")
    print("Done!")


if __name__ == "__main__":
    main()
