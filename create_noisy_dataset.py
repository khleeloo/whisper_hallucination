"""
Create noisy label-mismatch datasets for Whisper hallucination experiments.

Noise patterns:
- UU (Unique-Unique): N unique audios paired with N unique unrelated sentences
- RR (Repeat-Repeat): K audio-text mismatch pairs, repeated to fill N
- RU (Repeat-Unique): K repeated audios, each time with a different unique sentence
- UR (Unique-Repeat): N unique audios paired with K repeated sentences

Design:
- Noisy datasets are created by substitution, not addition.
- Total train.tsv size stays equal to the clean baseline size.
- No noisy-source audio is also kept with its original clean label.
- Supports 8%, 16%, 32%, 64%, and 100% noise.
"""

import argparse
import csv
import os
import random
import shutil
from typing import Dict, List, Sequence, Set, Tuple


Row = Dict[str, str]


def load_tsv(tsv_path: str) -> Tuple[List[str], List[Row]]:
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Missing TSV file: {tsv_path}")

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No header found in TSV file: {tsv_path}")
        rows = list(reader)

    if "sentence" not in fieldnames:
        raise ValueError(f"Expected a 'sentence' column in {tsv_path}")

    return fieldnames, rows


def write_tsv(tsv_path: str, fieldnames: Sequence[str], rows: Sequence[Row]) -> None:
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows to {tsv_path}", flush=True)


def create_subset(rows: Sequence[Row], subset_size: int, rng: random.Random) -> List[Row]:
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")

    if subset_size >= len(rows):
        print(
            f"  Requested subset_size {subset_size} >= total {len(rows)}, using all data",
            flush=True,
        )
        return [dict(row) for row in rows]

    return [dict(row) for row in rng.sample(list(rows), subset_size)]


def check_basic_args(clean_rows: Sequence[Row], n_noisy: int, n_repeat: int) -> None:
    n_total = len(clean_rows)

    if n_total == 0:
        raise ValueError("No clean rows available.")

    if n_noisy < 0:
        raise ValueError(f"n_noisy must be non-negative, got {n_noisy}")

    if n_noisy > n_total:
        raise ValueError(
            f"n_noisy={n_noisy} cannot exceed clean dataset size={n_total}. "
            "Use noise_ratio <= 1.0."
        )

    if n_repeat <= 0:
        raise ValueError(f"n_repeat_pairs must be positive, got {n_repeat}")

    if n_repeat > n_total:
        raise ValueError(
            f"n_repeat_pairs={n_repeat} cannot exceed clean dataset size={n_total}"
        )


def make_pairwise_mismatched_assignment(
    source_indices: Sequence[int],
    sentence_indices: Sequence[int],
    rng: random.Random,
    n_total_rows: int,
    max_attempts: int = 1000,
) -> List[int]:
    """
    Assign sentence indices to source audio indices with pairwise mismatch:
    assigned_sentence[i] != source_indices[i].

    This supports 100% noise. For UU at 100%, this creates a derangement.
    """
    source_indices = list(source_indices)
    sentence_indices = list(sentence_indices)

    if len(source_indices) != len(sentence_indices):
        raise ValueError(
            f"source_indices and sentence_indices must have same length, got "
            f"{len(source_indices)} and {len(sentence_indices)}"
        )

    n = len(source_indices)

    if n == 0:
        return []

    if n_total_rows <= 1:
        raise ValueError("Need at least 2 rows to create mismatched labels.")

    if n == 1:
        if sentence_indices[0] != source_indices[0]:
            return sentence_indices

        candidates = [idx for idx in range(n_total_rows) if idx != source_indices[0]]
        return [rng.choice(candidates)]

    assigned = list(sentence_indices)

    for _ in range(max_attempts):
        rng.shuffle(assigned)
        if all(src_idx != sent_idx for src_idx, sent_idx in zip(source_indices, assigned)):
            return assigned

    # Fallback local repair.
    assigned = list(sentence_indices)
    rng.shuffle(assigned)

    for i in range(n):
        if assigned[i] != source_indices[i]:
            continue

        swap_found = False
        for j in range(n):
            if i == j:
                continue

            if assigned[j] != source_indices[i] and assigned[i] != source_indices[j]:
                assigned[i], assigned[j] = assigned[j], assigned[i]
                swap_found = True
                break

        if not swap_found:
            raise RuntimeError(
                "Failed to construct pairwise mismatched assignment. "
                "Try a different seed."
            )

    if not all(src_idx != sent_idx for src_idx, sent_idx in zip(source_indices, assigned)):
        raise RuntimeError("Internal error: assignment still contains self-pairs.")

    return assigned


def create_noise_uu(
    clean_rows: Sequence[Row],
    n_noisy: int,
    rng: random.Random,
) -> Tuple[List[Row], Set[int]]:
    """
    Unique-Unique:
    N unique audios paired with N unique unrelated sentences.

    Supports 100% noise by using pairwise mismatch instead of requiring
    globally disjoint audio/sentence pools.
    """
    if n_noisy == 0:
        return [], set()

    n_total = len(clean_rows)

    source_indices = rng.sample(range(n_total), n_noisy)
    sentence_indices = rng.sample(range(n_total), n_noisy)

    sentence_indices = make_pairwise_mismatched_assignment(
        source_indices=source_indices,
        sentence_indices=sentence_indices,
        rng=rng,
        n_total_rows=n_total,
    )

    noisy_rows = []
    for src_idx, sent_idx in zip(source_indices, sentence_indices):
        row = dict(clean_rows[src_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        noisy_rows.append(row)

    rng.shuffle(noisy_rows)
    return noisy_rows, set(source_indices)


def create_noise_rr(
    clean_rows: Sequence[Row],
    n_noisy: int,
    rng: random.Random,
    n_pairs: int = 10,
) -> Tuple[List[Row], Set[int]]:
    """
    Repeat-Repeat:
    K mismatched audio-text pairs, each repeated to fill N noisy rows.
    """
    if n_noisy == 0:
        return [], set()

    n_total = len(clean_rows)
    n_pairs = min(n_pairs, n_noisy)

    audio_indices = rng.sample(range(n_total), n_pairs)
    sentence_indices = rng.sample(range(n_total), n_pairs)

    sentence_indices = make_pairwise_mismatched_assignment(
        source_indices=audio_indices,
        sentence_indices=sentence_indices,
        rng=rng,
        n_total_rows=n_total,
    )

    base_pairs = []
    for audio_idx, sent_idx in zip(audio_indices, sentence_indices):
        row = dict(clean_rows[audio_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        base_pairs.append(row)

    noisy_rows = []
    repeats_per_pair = n_noisy // n_pairs
    remainder = n_noisy % n_pairs

    for i, pair in enumerate(base_pairs):
        count = repeats_per_pair + (1 if i < remainder else 0)
        noisy_rows.extend([dict(pair) for _ in range(count)])

    rng.shuffle(noisy_rows)
    return noisy_rows, set(audio_indices)


def create_noise_ru(
    clean_rows: Sequence[Row],
    n_noisy: int,
    rng: random.Random,
    n_audios: int = 10,
) -> Tuple[List[Row], Set[int]]:
    """
    Repeat-Unique:
    K repeated audios, each time paired with a different unique sentence.

    Supports 100% noise by enforcing pairwise mismatch only.
    """
    if n_noisy == 0:
        return [], set()

    n_total = len(clean_rows)
    n_audios = min(n_audios, n_noisy)

    audio_indices = rng.sample(range(n_total), n_audios)
    audio_sequence = [audio_indices[i % n_audios] for i in range(n_noisy)]

    # Unique sentence labels. At 100% noise this uses every clean sentence once.
    sentence_indices = rng.sample(range(n_total), n_noisy)

    sentence_indices = make_pairwise_mismatched_assignment(
        source_indices=audio_sequence,
        sentence_indices=sentence_indices,
        rng=rng,
        n_total_rows=n_total,
    )

    noisy_rows = []
    for audio_idx, sent_idx in zip(audio_sequence, sentence_indices):
        if sent_idx == audio_idx:
            raise RuntimeError("Internal error: RU produced a self-pair.")

        row = dict(clean_rows[audio_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        noisy_rows.append(row)

    rng.shuffle(noisy_rows)
    return noisy_rows, set(audio_indices)


def create_noise_ur(
    clean_rows: Sequence[Row],
    n_noisy: int,
    rng: random.Random,
    n_sentences: int = 10,
) -> Tuple[List[Row], Set[int]]:
    """
    Unique-Repeat:
    N unique audios paired with K repeated unrelated sentences.

    Supports 100% noise by allowing all audios as sources while ensuring
    no audio receives its own original sentence.
    """
    if n_noisy == 0:
        return [], set()

    n_total = len(clean_rows)
    n_sentences = min(n_sentences, n_noisy)

    if n_sentences < 2:
        raise ValueError("UR needs at least 2 repeated sentences to avoid self-pairs.")

    sentence_indices = rng.sample(range(n_total), n_sentences)
    audio_indices = rng.sample(range(n_total), n_noisy)

    assigned_sentence_indices = []

    for i, audio_idx in enumerate(audio_indices):
        sent_idx = sentence_indices[i % n_sentences]

        if sent_idx == audio_idx:
            replacement = None
            for offset in range(1, n_sentences):
                candidate = sentence_indices[(i + offset) % n_sentences]
                if candidate != audio_idx:
                    replacement = candidate
                    break

            if replacement is None:
                raise RuntimeError(
                    "Failed to avoid self-pair in UR. Increase n_repeat_pairs."
                )

            sent_idx = replacement

        assigned_sentence_indices.append(sent_idx)

    noisy_rows = []
    for audio_idx, sent_idx in zip(audio_indices, assigned_sentence_indices):
        if sent_idx == audio_idx:
            raise RuntimeError("Internal error: UR produced a self-pair.")

        row = dict(clean_rows[audio_idx])
        row["sentence"] = clean_rows[sent_idx]["sentence"]
        noisy_rows.append(row)

    rng.shuffle(noisy_rows)
    return noisy_rows, set(audio_indices)


def get_audio_key(row: Row) -> str:
    """
    Return a stable audio key for sanity checks.
    Common Voice usually uses the 'path' column.
    """
    for key in ("path", "audio", "audio_path", "filename", "file"):
        if key in row and row[key]:
            return row[key]
    return repr(row)


def substitute_noisy_rows(
    clean_rows: Sequence[Row],
    noisy_rows: Sequence[Row],
    used_noisy_audio_indices: Set[int],
    rng: random.Random,
) -> List[Row]:
    """
    Replace clean rows with noisy rows while keeping total dataset size fixed.

    Any audio used as a noisy-source audio is excluded from the clean kept subset.
    """
    n_total = len(clean_rows)
    n_noisy = len(noisy_rows)
    n_keep = n_total - n_noisy

    clean_candidates = [
        dict(row)
        for idx, row in enumerate(clean_rows)
        if idx not in used_noisy_audio_indices
    ]

    if n_keep > len(clean_candidates):
        raise ValueError(
            f"Cannot keep {n_keep} clean rows after excluding "
            f"{len(used_noisy_audio_indices)} noisy-source audios. "
            f"Only {len(clean_candidates)} clean candidates remain."
        )

    kept_clean = rng.sample(clean_candidates, n_keep)
    combined = kept_clean + [dict(row) for row in noisy_rows]
    rng.shuffle(combined)

    if len(combined) != n_total:
        raise RuntimeError(
            f"Combined dataset has wrong size: {len(combined)} != {n_total}"
        )

    return combined


def sanity_check_no_clean_duplicate_for_noisy_sources(
    clean_rows: Sequence[Row],
    combined_rows: Sequence[Row],
    used_noisy_audio_indices: Set[int],
    noisy_rows: Sequence[Row],
) -> None:
    """
    Warn if a noisy-source audio appears more times than expected.

    RR/RU intentionally repeat noisy audios, so expected counts are computed
    from noisy_rows.
    """
    if not used_noisy_audio_indices:
        return

    noisy_source_keys = {
        get_audio_key(clean_rows[idx]) for idx in used_noisy_audio_indices
    }

    expected_counts = {}
    for row in noisy_rows:
        key = get_audio_key(row)
        expected_counts[key] = expected_counts.get(key, 0) + 1

    actual_counts = {}
    for row in combined_rows:
        key = get_audio_key(row)
        if key in noisy_source_keys:
            actual_counts[key] = actual_counts.get(key, 0) + 1

    suspicious = []
    for key, actual_count in actual_counts.items():
        expected_count = expected_counts.get(key, 0)
        if actual_count != expected_count:
            suspicious.append((key, actual_count, expected_count))

    if suspicious:
        print("  WARNING: possible clean duplicate of noisy-source audio detected:")
        for key, actual_count, expected_count in suspicious[:10]:
            print(
                f"    {key}: actual_count={actual_count}, "
                f"expected_noisy_count={expected_count}",
                flush=True,
            )
        if len(suspicious) > 10:
            print(f"    ... plus {len(suspicious) - 10} more", flush=True)


def copy_eval_splits(data_dir: str, output_dir: str) -> None:
    for split in ["dev.tsv", "test.tsv"]:
        src = os.path.join(data_dir, split)
        dst = os.path.join(output_dir, split)

        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required split: {src}")

        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {split} to {output_dir}", flush=True)
        else:
            print(f"  {split} already exists in {output_dir}; not overwriting", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create noisy datasets for Whisper hallucination experiments"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to Common Voice language directory, e.g. .../en",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for noisy datasets",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=120000,
        help="Number of clean training utterances to use",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.08,
        help="Fraction of training rows to replace with noisy rows, e.g. 0.08, 0.16, 0.32, 0.64, 1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_repeat_pairs",
        type=int,
        default=10,
        help="Number of repeated pairs/audios/sentences for RR/RU/UR patterns",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output train.tsv/noisy_only.tsv files",
    )
    args = parser.parse_args()

    if not (0.0 <= args.noise_ratio <= 1.0):
        raise ValueError(f"noise_ratio must be in [0, 1], got {args.noise_ratio}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_tsv = os.path.join(args.data_dir, "train.tsv")
    print(f"Loading {train_tsv}...", flush=True)
    fieldnames, all_rows = load_tsv(train_tsv)
    print(f"  Total training utterances: {len(all_rows)}", flush=True)

    subset_rng = random.Random(args.seed)

    print(f"Creating subset of {args.subset_size} utterances...", flush=True)
    clean_rows = create_subset(all_rows, args.subset_size, subset_rng)

    n_noisy = int(len(clean_rows) * args.noise_ratio)
    check_basic_args(clean_rows, n_noisy, args.n_repeat_pairs)

    print(f"  Clean subset size: {len(clean_rows)}", flush=True)
    print(f"  Noisy utterances per config: {n_noisy}", flush=True)

    baseline_dir = os.path.join(args.output_dir, "base")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_train_path = os.path.join(baseline_dir, "train.tsv")

    if os.path.exists(baseline_train_path) and not args.overwrite:
        print(f"  Baseline already exists; not overwriting: {baseline_train_path}", flush=True)
    else:
        write_tsv(baseline_train_path, fieldnames, clean_rows)

    copy_eval_splits(args.data_dir, args.output_dir)

    noise_configs = {
        "uu": ("Unique-Unique", create_noise_uu, 101),
        "rr": ("Repeat-Repeat", create_noise_rr, 202),
        "ru": ("Repeat-Unique", create_noise_ru, 303),
        "ur": ("Unique-Repeat", create_noise_ur, 404),
    }

    summary_lines = []

    for config_name, (description, noise_fn, seed_offset) in noise_configs.items():
        print(f"\nCreating {config_name} ({description}) noise...", flush=True)

        config_rng = random.Random(args.seed + seed_offset)

        if config_name == "rr":
            noisy_rows, used_noisy_audio_indices = noise_fn(
                clean_rows,
                n_noisy,
                config_rng,
                n_pairs=args.n_repeat_pairs,
            )
        elif config_name == "ru":
            noisy_rows, used_noisy_audio_indices = noise_fn(
                clean_rows,
                n_noisy,
                config_rng,
                n_audios=args.n_repeat_pairs,
            )
        elif config_name == "ur":
            noisy_rows, used_noisy_audio_indices = noise_fn(
                clean_rows,
                n_noisy,
                config_rng,
                n_sentences=args.n_repeat_pairs,
            )
        else:
            noisy_rows, used_noisy_audio_indices = noise_fn(
                clean_rows,
                n_noisy,
                config_rng,
            )

        combined = substitute_noisy_rows(
            clean_rows=clean_rows,
            noisy_rows=noisy_rows,
            used_noisy_audio_indices=used_noisy_audio_indices,
            rng=config_rng,
        )

        sanity_check_no_clean_duplicate_for_noisy_sources(
            clean_rows=clean_rows,
            combined_rows=combined,
            used_noisy_audio_indices=used_noisy_audio_indices,
            noisy_rows=noisy_rows,
        )

        config_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        train_out = os.path.join(config_dir, "train.tsv")
        noisy_out = os.path.join(config_dir, "noisy_only.tsv")

        if os.path.exists(train_out) and not args.overwrite:
            raise FileExistsError(
                f"{train_out} already exists. Use --overwrite if you want to replace it."
            )

        if os.path.exists(noisy_out) and not args.overwrite:
            raise FileExistsError(
                f"{noisy_out} already exists. Use --overwrite if you want to replace it."
            )

        write_tsv(train_out, fieldnames, combined)
        write_tsv(noisy_out, fieldnames, noisy_rows)

        n_keep = len(clean_rows) - len(noisy_rows)
        unique_noisy_source_audios = len(used_noisy_audio_indices)

        msg = (
            f"  {config_name}: {n_keep} clean + {len(noisy_rows)} noisy "
            f"= {len(combined)} total "
            f"({args.noise_ratio * 100:.1f}% noise, same size); "
            f"unique noisy-source audios={unique_noisy_source_audios}"
        )
        print(msg, flush=True)
        summary_lines.append(msg)

    meta_path = os.path.join(args.output_dir, "experiment_info.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {args.data_dir}\n")
        f.write(f"Subset size: {len(clean_rows)}\n")
        f.write(f"Noise ratio: {args.noise_ratio}\n")
        f.write(f"Noisy utterances per config: {n_noisy}\n")
        f.write(f"Repeat pairs/audios/sentences: {args.n_repeat_pairs}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("Configs: base, uu, rr, ru, ur\n")
        f.write("\nDesign notes:\n")
        f.write("- Noisy datasets use substitution, not addition.\n")
        f.write("- Total train.tsv size equals the clean baseline size.\n")
        f.write("- Noisy-source audios are excluded from the kept clean subset.\n")
        f.write("- Supports 100% noise with pairwise mismatch constraints.\n")
        f.write("\nSummary:\n")
        for line in summary_lines:
            f.write(line.strip() + "\n")

    print(f"\nExperiment info saved to {meta_path}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()