#!/usr/bin/env python3
"""Prepare corrected paired HALAS annotation set from an already completed inference run.

Uses the released HALAS Whisper-large-v3 transcript as baseline and the locally generated
fine-tuned transcript. No ASR inference is performed.

Outputs:
  annotation_paired_private_manifest.csv
  annotation_paired_blinded.csv
  paired_generation_summary.json

The public file contains 600 randomized rows for 300 audio pairs (baseline + fine-tuned),
with model condition hidden from annotators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_csv",
        type=Path,
        default=Path(
            "/scratch/vemotionsys/rmfrieske/whisper_hallucination/"
            "halas_finetune_forgetting/halas_whisper_v3_pretrained_vs_finetuned.csv"
        ),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/scratch/vemotionsys/rmfrieske/whisper_hallucination/halas_finetune_forgetting"
        ),
    )
    p.add_argument("--sample_size", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260828)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {
        "audio_id",
        "audio_path",
        "halas_prediction",
        "finetuned_prediction",
        "halas_human_hallucination",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in completed HALAS inference CSV: {missing}")

    usable = df[
        df["halas_prediction"].fillna("").astype(str).str.strip().ne("")
        & df["finetuned_prediction"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if len(usable) < args.sample_size:
        raise ValueError(f"Only {len(usable)} usable paired rows, need {args.sample_size}")

    sample = usable.sample(n=args.sample_size, random_state=args.seed).copy().reset_index(drop=True)
    sample["pair_id"] = [f"halas_pair_{i:04d}" for i in range(len(sample))]

    private_rows = []
    public_rows = []
    for row in sample.itertuples(index=False):
        for condition, transcript in [
            ("halas_pretrained", row.halas_prediction),
            ("finetuned", row.finetuned_prediction),
        ]:
            private_rows.append(
                {
                    "pair_id": row.pair_id,
                    "audio_id": row.audio_id,
                    "audio_path": row.audio_path,
                    "condition": condition,
                    "transcript": transcript,
                    "halas_original_human_hallucination": int(bool(row.halas_human_hallucination)),
                }
            )

    private = pd.DataFrame(private_rows)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(private))
    private = private.iloc[order].reset_index(drop=True)
    private["annotation_id"] = [f"halas_ann_{i:04d}" for i in range(len(private))]

    public = private[["annotation_id", "pair_id", "audio_path", "transcript"]].copy()
    public["human_hallucination"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_path = args.output_dir / "annotation_paired_private_manifest.csv"
    public_path = args.output_dir / "annotation_paired_blinded.csv"
    private.to_csv(private_path, index=False)
    public.to_csv(public_path, index=False)

    summary = {
        "source_csv": str(args.input_csv),
        "available_pairs": int(len(usable)),
        "sample_pairs": int(args.sample_size),
        "annotation_rows": int(len(public)),
        "conditions_per_pair": 2,
        "seed": int(args.seed),
        "baseline": "released HALAS whisper_large_v3 transcript",
        "comparison": "clean Common Voice LoRA fine-tuned Whisper-large-v3",
        "note": "No local pretrained re-decoding is used; both transcripts in each sampled pair should be annotated under the same blinded protocol.",
    }
    (args.output_dir / "paired_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Private manifest: {private_path}")
    print(f"Blinded annotation file: {public_path}")


if __name__ == "__main__":
    main()
