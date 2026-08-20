#!/usr/bin/env python3
"""Run the paper-facing before/after acoustic abstention experiment.

This driver intentionally keeps the mitigation experiment narrow:

  * model: fixed Base Whisper checkpoint
  * DEV + TEST: disjoint splits
  * conditions: clean, full-noise 0.50, full-noise 0.75
  * hallucination labels: Qwen3 primary + GPT-2 robustness, thresholds from
    clean Base DEV only
  * mitigation: reference-free wav2vec2-CTC acoustic-consistency gate
  * gate threshold: selected on DEV only, subject to >=98% clean DEV coverage
  * TEST: threshold frozen; no retuning

The underlying implementation lives in ``acoustic_abstention_mitigation.py``.
This file fixes the paper protocol and additionally writes a compact
``paper_before_after_summary.csv`` with the quantities needed in the manuscript:
coverage, hallucination incidence before gating, emitted hallucination incidence
after gating, hallucination risk among emitted transcripts, capture recall, and
WER before/after gating.

Important interpretation:
``after_system_hallucination_rate`` is the fraction of all input utterances for
which the gated system still emits a hallucination-like transcript. Abstentions
are therefore counted as safe non-emissions. ``hallucination_rate_among_emitted``
is reported separately so mitigation cannot look successful merely by reducing
coverage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_OUTPUT_DIR = ROOT / "hallucination_mitigation_acoustic_before_after"
DEFAULT_MODEL_DIR = ROOT / "base" / "checkpoint-10000"
DEFAULT_TEST_TSV = Path("/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv")
DEFAULT_CLIPS_DIR = Path(
    "/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
)
DEFAULT_PERTURBATIONS = [
    "none",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]


def build_paper_summary(test_summary: pd.DataFrame) -> pd.DataFrame:
    """Convert the detailed mitigation summary to a paper-facing table."""
    required = {
        "perturbation",
        "N",
        "coverage",
        "abstention_rate",
        "mean_WER_all",
        "mean_WER_accepted",
        "hallucination_rate_before_qwen",
        "emitted_hallucination_incidence_qwen",
        "residual_hallucination_among_accepted_qwen",
        "hallucination_capture_recall_qwen",
        "hallucination_rate_before_gpt2",
        "emitted_hallucination_incidence_gpt2",
        "residual_hallucination_among_accepted_gpt2",
        "hallucination_capture_recall_gpt2",
    }
    missing = sorted(required - set(test_summary.columns))
    if missing:
        raise ValueError(f"TEST mitigation summary is missing columns: {missing}")

    order = {name: i for i, name in enumerate(DEFAULT_PERTURBATIONS)}
    out = test_summary.copy()
    out["_order"] = out["perturbation"].map(order).fillna(len(order))
    out = out.sort_values(["_order", "perturbation"]).drop(columns="_order")

    compact = pd.DataFrame(
        {
            "condition": out["perturbation"],
            "N": out["N"].astype(int),
            "coverage_pct": 100.0 * out["coverage"],
            "abstention_pct": 100.0 * out["abstention_rate"],
            "WER_before": out["mean_WER_all"],
            "WER_among_emitted": out["mean_WER_accepted"],
            "Qwen_H_before_pct": 100.0 * out["hallucination_rate_before_qwen"],
            "Qwen_H_after_system_pct": 100.0 * out["emitted_hallucination_incidence_qwen"],
            "Qwen_H_among_emitted_pct": 100.0 * out[
                "residual_hallucination_among_accepted_qwen"
            ],
            "Qwen_H_capture_pct": 100.0 * out["hallucination_capture_recall_qwen"],
            "GPT2_H_before_pct": 100.0 * out["hallucination_rate_before_gpt2"],
            "GPT2_H_after_system_pct": 100.0 * out["emitted_hallucination_incidence_gpt2"],
            "GPT2_H_among_emitted_pct": 100.0 * out[
                "residual_hallucination_among_accepted_gpt2"
            ],
            "GPT2_H_capture_pct": 100.0 * out["hallucination_capture_recall_gpt2"],
        }
    )
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen DEV->TEST acoustic-consistency abstention experiment."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dev_tsv", type=Path, default=None)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--base_model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dev_max_samples", type=int, default=1000)
    parser.add_argument("--test_max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ctc_batch_size", type=int, default=8)
    parser.add_argument("--lm_batch_size", type=int, default=8)
    parser.add_argument("--min_clean_coverage", type=float, default=0.98)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpt2_model", default="gpt2")
    parser.add_argument("--wav2vec2_model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse_generated_outputs", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.min_clean_coverage <= 1.0):
        raise ValueError("--min_clean_coverage must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = Path(__file__).with_name("acoustic_abstention_mitigation.py")
    if not core.exists():
        raise FileNotFoundError(f"Missing core experiment script: {core}")

    cmd = [
        sys.executable,
        "-u",
        str(core),
        "--output_dir",
        str(args.output_dir),
        "--test_tsv",
        str(args.test_tsv),
        "--clips_dir",
        str(args.clips_dir),
        "--base_model_dir",
        str(args.base_model_dir),
        "--qwen_model",
        args.qwen_model,
        "--gpt2_model",
        args.gpt2_model,
        "--dev_max_samples",
        str(args.dev_max_samples),
        "--test_max_samples",
        str(args.test_max_samples),
        "--batch_size",
        str(args.batch_size),
        "--ctc_batch_size",
        str(args.ctc_batch_size),
        "--lm_batch_size",
        str(args.lm_batch_size),
        "--min_clean_coverage",
        str(args.min_clean_coverage),
        "--bootstrap",
        str(args.bootstrap),
        "--seed",
        str(args.seed),
        "--perturbations",
        *DEFAULT_PERTURBATIONS,
    ]
    if args.dev_tsv is not None:
        cmd.extend(["--dev_tsv", str(args.dev_tsv)])
    if args.wav2vec2_model is not None:
        cmd.extend(["--wav2vec2_model", args.wav2vec2_model])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.reuse_generated_outputs:
        cmd.append("--reuse_generated_outputs")

    print("=== Paper protocol: acoustic abstention before/after ===", flush=True)
    print("Conditions: clean, full-noise 0.50, full-noise 0.75", flush=True)
    print(
        f"Gate selection: DEV only; clean coverage >= {100*args.min_clean_coverage:.1f}%",
        flush=True,
    )
    print("TEST threshold is frozen; Qwen3 primary, GPT-2 parallel.", flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    test_summary_path = args.output_dir / "test_mitigation_summary.csv"
    if not test_summary_path.exists():
        raise FileNotFoundError(f"Core experiment did not create {test_summary_path}")

    test_summary = pd.read_csv(test_summary_path)
    paper_summary = build_paper_summary(test_summary)
    paper_path = args.output_dir / "paper_before_after_summary.csv"
    paper_summary.to_csv(paper_path, index=False)

    print("\n=== PAPER-FACING HELD-OUT TEST BEFORE/AFTER ===", flush=True)
    print(paper_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\nSaved: {paper_path}", flush=True)


if __name__ == "__main__":
    main()
