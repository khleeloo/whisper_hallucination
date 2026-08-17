"""Merge Whisper eval_validation per-utterance CSVs for acoustic validation.

The eval_validation directory may contain one CSV per model/checkpoint. These
files already include `audio_path`, references, hypotheses, WER, GPT2 scores,
Qwen scores, conditions, and repetition metrics. This script concatenates them
into one input CSV for acoustic_grounding_validation.py.

Example:
    python prepare_eval_validation_acoustic_input.py \
        --input_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation \
        --output_csv /scratch/vemotionsys/rmfrieske/whisper_hallucination/acoustic_validation/eval_validation_acoustic_input.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT_DIR = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation")
DEFAULT_OUTPUT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination/acoustic_validation/eval_validation_acoustic_input.csv")


def input_paths(input_dir: Path) -> List[Path]:
    paths = sorted(input_dir.glob("per_utterance_*.csv"))
    return [path for path in paths if path.name != "per_utterance_metrics_whisper.csv"]


def merge_eval_validation(input_dir: Path) -> pd.DataFrame:
    paths = input_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No per_utterance_*.csv files found in {input_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        required = {"utt_id", "audio_path", "reference", "hypothesis", "wer"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
        df = df.copy()
        df["source_eval_csv"] = str(path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    missing_audio = merged["audio_path"].isna() | merged["audio_path"].astype(str).str.len().eq(0)
    if bool(missing_audio.any()):
        raise ValueError(f"Merged CSV has {int(missing_audio.sum())} rows without audio_path")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge eval_validation per-utterance CSVs for acoustic validation.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_eval_validation(args.input_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Merged {len(input_paths(args.input_dir))} files and {len(merged)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
