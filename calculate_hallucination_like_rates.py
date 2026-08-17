#!/usr/bin/env python3
"""Compute WER/Qwen hallucination-like rates from per-utterance CSVs.

The aggregate validation table is not enough to recover a hallucination-like
rate. This script reads the saved per-utterance CSVs and applies the canonical
baseline-threshold definition:

    wer_i > mean_base_wer and fluency_i > mean_base_fluency

By default it evaluates the current 64% table models plus all available RR/RU
64% checkpoints in eval_64pct. The primary hallucination_like_rate column uses
Qwen3-0.6B, matching the LM-rescored fairseq evaluator's primary LM selection.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_BASE_FILE = SCRATCH_ROOT / "eval_validation" / "per_utterance_base_ckpt14000.csv"
DEFAULT_EVAL_64PCT_DIR = SCRATCH_ROOT / "eval_64pct"
DEFAULT_OUTPUT_TSV = Path("plots/hallucination_like_rates_64pct.tsv")
DEFAULT_COMPARISON_CSV = Path("results/mitigation/wacc_vs_wer_criterion_comparison.csv")


def condition_from_model_name(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("base"):
        return "base"
    if lowered.startswith("rr"):
        return "RR"
    if lowered.startswith("ru"):
        return "RU"
    if lowered.startswith("uu"):
        return "UU"
    if lowered.startswith("ur"):
        return "UR"
    return model_name


def checkpoint_from_model_name(model_name: str) -> str:
    if model_name == "base_ckpt14000":
        return "base_ckpt14000"
    match = re.search(r"(checkpoint-\d+|final)", model_name)
    return match.group(1) if match else ""


def training_noise_pct_from_model_name(model_name: str) -> int:
    return 0 if model_name.startswith("base") else 64


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def default_model_files(eval_64pct_dir: Path, base_file: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {"base_ckpt14000": [base_file]}
    for path in sorted(eval_64pct_dir.glob("per_utterance_*.csv")):
        stem = path.stem.replace("per_utterance_", "")
        model_name = re.sub(r"_shard\d+-of-\d+$", "", stem)
        files.setdefault(model_name, []).append(path)
    return files


def compute_rate(df: pd.DataFrame, wer_threshold: float, fluency_col: str, fluency_threshold: float) -> tuple[int, float]:
    mask = (df["wer"] > wer_threshold) & (df[fluency_col] > fluency_threshold)
    count = int(mask.sum())
    rate = float(mask.mean()) if len(mask) else 0.0
    return count, rate


def compute_old_wacc_rate(df: pd.DataFrame, wacc_threshold: float, fluency_col: str, fluency_threshold: float) -> pd.Series:
    return (df["wacc"] < wacc_threshold) & (df[fluency_col] > fluency_threshold)


def compute_new_wer_rate(df: pd.DataFrame, wer_threshold: float, fluency_col: str, fluency_threshold: float) -> pd.Series:
    return (df["wer"] > wer_threshold) & (df[fluency_col] > fluency_threshold)


def build_rates(model_files: dict[str, list[Path]], base_file: Path) -> pd.DataFrame:
    base_df = pd.read_csv(base_file)
    required_cols = {"wer", "normalized_sentence_score_gpt2", "normalized_sentence_score_Qwen3-0.6B"}
    missing = required_cols.difference(base_df.columns)
    if missing:
        raise ValueError(f"Base CSV missing required columns: {sorted(missing)}")

    wer_threshold = float(base_df["wer"].mean())
    gpt2_threshold = float(base_df["normalized_sentence_score_gpt2"].mean())
    qwen_threshold = float(base_df["normalized_sentence_score_Qwen3-0.6B"].mean())

    rows = []
    for model_name, paths in sorted(model_files.items()):
        df = load_csvs(paths)
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{model_name} CSV missing required columns: {sorted(missing)}")

        qwen_count, qwen_rate = compute_rate(
            df,
            wer_threshold,
            "normalized_sentence_score_Qwen3-0.6B",
            qwen_threshold,
        )
        gpt2_count, gpt2_rate = compute_rate(
            df,
            wer_threshold,
            "normalized_sentence_score_gpt2",
            gpt2_threshold,
        )

        rows.append({
            "Condition": condition_from_model_name(model_name),
            "Training_noise_pct": training_noise_pct_from_model_name(model_name),
            "Config": model_name.replace("_checkpoint-", "_checkpoint-"),
            "Model_checkpoint": checkpoint_from_model_name(model_name),
            "Num_utterances": len(df),
            "hallucination_like_count": qwen_count,
            "hallucination_like_rate": qwen_rate,
            "Hallucination_like_count_gpt2": gpt2_count,
            "Hallucination_like_rate_gpt2": gpt2_rate,
            "primary_lm": "Qwen3-0.6B",
            "WER_threshold_base_mean": wer_threshold,
            "Fluency_Qwen3_0_6B_threshold_base_mean": qwen_threshold,
            "Fluency_gpt2_threshold_base_mean": gpt2_threshold,
            "Source_files": ";".join(str(path) for path in paths),
        })

    return pd.DataFrame(rows)


def build_wacc_vs_wer_comparison(model_files: dict[str, list[Path]], base_file: Path) -> pd.DataFrame:
    base_df = pd.read_csv(base_file)
    required_cols = {"wer", "wacc", "normalized_sentence_score_Qwen3-0.6B"}
    missing = required_cols.difference(base_df.columns)
    if missing:
        raise ValueError(f"Base CSV missing required columns: {sorted(missing)}")

    old_wacc_threshold = float(base_df["wacc"].mean())
    new_wer_threshold = float(base_df["wer"].mean())
    qwen_threshold = float(base_df["normalized_sentence_score_Qwen3-0.6B"].mean())

    rows = []
    for model_name, paths in sorted(model_files.items()):
        df = load_csvs(paths)
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{model_name} CSV missing required columns: {sorted(missing)}")
        old_mask = compute_old_wacc_rate(
            df,
            old_wacc_threshold,
            "normalized_sentence_score_Qwen3-0.6B",
            qwen_threshold,
        )
        new_mask = compute_new_wer_rate(
            df,
            new_wer_threshold,
            "normalized_sentence_score_Qwen3-0.6B",
            qwen_threshold,
        )
        old_rate = float(old_mask.mean()) if len(old_mask) else 0.0
        new_rate = float(new_mask.mean()) if len(new_mask) else 0.0
        rows.append({
            "Condition": condition_from_model_name(model_name),
            "Config": model_name.replace("_checkpoint-", "_checkpoint-"),
            "Model_checkpoint": checkpoint_from_model_name(model_name),
            "Num_utterances": len(df),
            "old_wacc_hallucination_like_count": int(old_mask.sum()),
            "old_wacc_hallucination_like_rate": old_rate,
            "new_wer_hallucination_like_count": int(new_mask.sum()),
            "new_wer_hallucination_like_rate": new_rate,
            "absolute_rate_difference": abs(new_rate - old_rate),
            "changed_label_count": int((old_mask != new_mask).sum()),
            "old_WAcc_threshold_base_mean": old_wacc_threshold,
            "new_WER_threshold_base_mean": new_wer_threshold,
            "Qwen3_threshold_base_mean": qwen_threshold,
            "Source_files": ";".join(str(path) for path in paths),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_file", type=Path, default=DEFAULT_BASE_FILE)
    parser.add_argument("--eval_64pct_dir", type=Path, default=DEFAULT_EVAL_64PCT_DIR)
    parser.add_argument("--output_tsv", type=Path, default=DEFAULT_OUTPUT_TSV)
    parser.add_argument("--comparison_csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument(
        "--current_table_only",
        action="store_true",
        help="Keep only base, RR checkpoint-9375, RU checkpoint-9375, UU final, and UR checkpoint-10000.",
    )
    args = parser.parse_args()

    model_files = default_model_files(args.eval_64pct_dir, args.base_file)
    if args.current_table_only:
        keep = {
            "base_ckpt14000",
            "rr_64pct_checkpoint-9375",
            "ru_64pct_checkpoint-9375",
            "uu_64pct_final",
            "ur_64pct_checkpoint-10000",
        }
        model_files = {name: paths for name, paths in model_files.items() if name in keep}

    rates = build_rates(model_files, args.base_file)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.output_tsv, sep="\t", index=False, float_format="%.6f")
    comparison = build_wacc_vs_wer_comparison(model_files, args.base_file)
    args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_csv, index=False, float_format="%.6f")

    display_cols = [
        "Condition",
        "Training_noise_pct",
        "Config",
        "Model_checkpoint",
        "Num_utterances",
        "hallucination_like_count",
        "hallucination_like_rate",
        "Hallucination_like_count_gpt2",
        "Hallucination_like_rate_gpt2",
    ]
    print(rates[display_cols].to_csv(sep="\t", index=False, float_format="%.6f"), end="")
    print(f"\nSaved: {args.output_tsv}")
    print(f"Saved: {args.comparison_csv}")


if __name__ == "__main__":
    main()