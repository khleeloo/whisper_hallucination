"""
Create noisy label-mismatch datasets for Whisper hallucination experiments.

Noise patterns (matching the paper):
- UU (Unique-Unique): N unique audios paired with N unique unrelated sentences
- RR (Repeat-Repeat): 10 audio-text mismatch pairs, repeated to fill N
- RU (Repeat-Unique): 10 repeated audios, each time with a different unique sentence
- UR (Unique-Repeat): N unique audios, all paired with 10 repeated sentences

Usage:
    python create_noisy_dataset.py \
        --data_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en \
        --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
        --subset_size 120000 \
        --noise_ratio 0.08 \
        --seed 42
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


def load_tsv(tsv_path):
    """Load a Common Voice TSV file, return header and rows."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def write_tsv(tsv_path, fieldnames, rows):
    """Write rows to a TSV file."""
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows to {tsv_path}")


def create_subset(rows, subset_size, rng):
    """Create a random subset of the training data."""
    if subset_size >= len(rows):
        print(f"  Requested subset_size {subset_size} >= total {len(rows)}, using all data")
        return list(rows)
    subset = rng.sample(rows, subset_size)
    return subset


def create_noise_uu(clean_rows, n_noisy, rng):
    """
    Unique-Unique: N unique audios paired with N unique unrelated sentences.
    Select N random rows, shuffle their sentences independently.
    """
    source_indices = rng.sample(range(len(clean_rows)), n_noisy)
    sentence_pool_indices = rng.sample(range(len(clean_rows)), n_noisy)

    # Ensure no sentence maps to its own audio
    for i in range(n_noisy):
        while sentence_pool_indices[i] == source_indices[i]:
            sentence_pool_indices[i] = rng.randint(0, len(clean_rows) - 1)

    noisy_rows = []
    for src_idx, sent_idx in zip(source_indices, sentence_pool_indices):
        row = dict(clean_rows[src_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        noisy_rows.append(row)

    return noisy_rows


def create_noise_rr(clean_rows, n_noisy, rng, n_pairs=10):
    """
    Repeat-Repeat: 10 mismatched audio-text pairs, each repeated ~N/10 times.
    """
    # Pick 10 random audios and 10 random (different) sentences
    audio_indices = rng.sample(range(len(clean_rows)), n_pairs)
    sentence_indices = rng.sample(range(len(clean_rows)), n_pairs)

    # Ensure no overlap
    for i in range(n_pairs):
        while sentence_indices[i] == audio_indices[i]:
            sentence_indices[i] = rng.randint(0, len(clean_rows) - 1)

    # Create the 10 base pairs
    base_pairs = []
    for audio_idx, sent_idx in zip(audio_indices, sentence_indices):
        row = dict(clean_rows[audio_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        base_pairs.append(row)

    # Repeat to reach n_noisy
    noisy_rows = []
    repeats_per_pair = n_noisy // n_pairs
    remainder = n_noisy % n_pairs
    for i, pair in enumerate(base_pairs):
        count = repeats_per_pair + (1 if i < remainder else 0)
        noisy_rows.extend([dict(pair) for _ in range(count)])

    rng.shuffle(noisy_rows)
    return noisy_rows


def create_noise_ru(clean_rows, n_noisy, rng, n_audios=10):
    """
    Repeat-Unique: 10 repeated audios, each time with a different unique sentence.
    """
    audio_indices = rng.sample(range(len(clean_rows)), n_audios)
    sentence_indices = rng.sample(range(len(clean_rows)), n_noisy)

    noisy_rows = []
    for i in range(n_noisy):
        audio_idx = audio_indices[i % n_audios]
        sent_idx = sentence_indices[i]
        # Ensure mismatch
        while sent_idx == audio_idx:
            sent_idx = rng.randint(0, len(clean_rows) - 1)
        row = dict(clean_rows[audio_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        noisy_rows.append(row)

    return noisy_rows


def create_noise_ur(clean_rows, n_noisy, rng, n_sentences=10):
    """
    Unique-Repeat: N unique audios, all paired with 10 repeated sentences.
    """
    audio_indices = rng.sample(range(len(clean_rows)), n_noisy)
    sentence_indices = rng.sample(range(len(clean_rows)), n_sentences)
    repeated_sentences = [clean_rows[idx]["sentence"] for idx in sentence_indices]

    noisy_rows = []
    for i, audio_idx in enumerate(audio_indices):
        row = dict(clean_rows[audio_idx])
        row["sentence"] = repeated_sentences[i % n_sentences]
        noisy_rows.append(row)

    return noisy_rows


def main():
    parser = argparse.ArgumentParser(description="Create noisy datasets for Whisper hallucination experiments")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Common Voice language directory (e.g., .../en)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for noisy datasets")
    parser.add_argument("--subset_size", type=int, default=120000,
                        help="Number of clean training utterances to use (to match LibriSpeech 360h scale)")
    parser.add_argument("--noise_ratio", type=float, default=0.08,
                        help="Fraction of noisy utterances to add (default: 0.08 = 8%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeat_pairs", type=int, default=10,
                        help="Number of repeated pairs/audios/sentences for RR/RU/UR patterns")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    train_tsv = os.path.join(args.data_dir, "train.tsv")
    print(f"Loading {train_tsv}...")
    fieldnames, all_rows = load_tsv(train_tsv)
    print(f"  Total training utterances: {len(all_rows)}")

    # Create subset
    print(f"Creating subset of {args.subset_size} utterances...")
    clean_rows = create_subset(all_rows, args.subset_size, rng)
    n_noisy = int(len(clean_rows) * args.noise_ratio)
    print(f"  Clean subset size: {len(clean_rows)}")
    print(f"  Noisy utterances per config: {n_noisy}")

    # Save clean subset as baseline training set
    baseline_dir = os.path.join(args.output_dir, "base")
    os.makedirs(baseline_dir, exist_ok=True)
    write_tsv(os.path.join(baseline_dir, "train.tsv"), fieldnames, clean_rows)

    # Copy dev and test sets
    for split in ["dev.tsv", "test.tsv"]:
        src = os.path.join(args.data_dir, split)
        dst_dir = os.path.join(args.output_dir, split)
        if not os.path.exists(dst_dir):
            shutil.copy2(src, os.path.join(args.output_dir, split))
            print(f"  Copied {split} to {args.output_dir}")

    # Create noisy configs
    noise_configs = {
        "uu": ("Unique-Unique", create_noise_uu),
        "rr": ("Repeat-Repeat", create_noise_rr),
        "ru": ("Repeat-Unique", create_noise_ru),
        "ur": ("Unique-Repeat", create_noise_ur),
    }

    for config_name, (description, noise_fn) in noise_configs.items():
        print(f"\nCreating {config_name} ({description}) noise...")
        if config_name == "rr":
            noisy_rows = noise_fn(clean_rows, n_noisy, rng, n_pairs=args.n_repeat_pairs)
        elif config_name == "ru":
            noisy_rows = noise_fn(clean_rows, n_noisy, rng, n_audios=args.n_repeat_pairs)
        elif config_name == "ur":
            noisy_rows = noise_fn(clean_rows, n_noisy, rng, n_sentences=args.n_repeat_pairs)
        else:
            noisy_rows = noise_fn(clean_rows, n_noisy, rng)

        # Substitute: replace a fraction of clean rows with noisy mismatches.
        # Dataset stays same total size as the clean baseline.
        n_keep = len(clean_rows) - n_noisy
        kept_clean = rng.sample(clean_rows, n_keep)
        combined = kept_clean + noisy_rows
        rng.shuffle(combined)

        config_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        write_tsv(os.path.join(config_dir, "train.tsv"), fieldnames, combined)

        # Save noisy subset separately for analysis
        write_tsv(os.path.join(config_dir, "noisy_only.tsv"), fieldnames, noisy_rows)

        print(f"  {config_name}: {n_keep} clean + {len(noisy_rows)} noisy = {len(combined)} total ({int(args.noise_ratio*100)}% noise, same size)")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "experiment_info.txt")
    with open(meta_path, "w") as f:
        f.write(f"Source: {args.data_dir}\n")
        f.write(f"Subset size: {len(clean_rows)}\n")
        f.write(f"Noise ratio: {args.noise_ratio}\n")
        f.write(f"Noisy utterances per config: {n_noisy}\n")
        f.write(f"Repeat pairs/audios/sentences: {args.n_repeat_pairs}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Configs: base, uu, rr, ru, ur\n")
    print(f"\nExperiment info saved to {meta_path}")
    print("Done!")


if __name__ == "__main__":
    main()
