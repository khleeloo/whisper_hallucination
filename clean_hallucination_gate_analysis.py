#!/usr/bin/env python3
"""Separate clean TEST hallucination-like outputs before/after the selected gate.

This analysis reuses ``scored_outputs.csv`` from the acoustic-abstention
experiments.  It does not rerun Whisper, Qwen3, GPT-2, or wav2vec2.

Default paper operating point
-----------------------------
The default "best gate" is the clean-DEV 95% coverage operating point.  The
threshold is recomputed from clean DEV CTC-NLL scores (never from TEST) using
the same rule as ``acoustic_abstention_clean_transfer.py`` and then frozen for
clean TEST.

Outputs
-------
The script writes one union file containing every clean TEST hallucination-like
output flagged by Qwen3 and/or GPT-2, plus separate before/caught/missed files
for each LM.  "Before" means all hallucination-like outputs before abstention;
"caught" means rejected by the gate; "after_missed" means hallucination-like
outputs that remain emitted after the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from acoustic_abstention_mitigation import accepted_mask
from acoustic_abstention_clean_transfer import threshold_for_min_coverage


ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_SOURCE = ROOT / "hallucination_mitigation_acoustic_before_after" / "scored_outputs.csv"
DEFAULT_OUTPUT_DIR = ROOT / "clean_hallucinations_best_gate"
DEFAULT_TARGET_COVERAGE = 0.95
CLEAN = "none"


def _require_columns(df: pd.DataFrame) -> None:
    required = {
        "split",
        "perturbation",
        "reference",
        "hypothesis",
        "WER",
        "qwen_plaus",
        "gpt2_plaus",
        "hallucination_like_qwen",
        "hallucination_like_gpt2",
        "ctc_support_nll",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in scored outputs: {missing}")


def _add_analysis_columns(clean_test: pd.DataFrame, tau: float) -> pd.DataFrame:
    out = clean_test.copy()
    out["accepted_best_gate"] = accepted_mask(out["ctc_support_nll"].astype(float), tau)
    out["abstained_best_gate"] = ~out["accepted_best_gate"]
    out["hallucination_union"] = (
        out["hallucination_like_qwen"].astype(bool)
        | out["hallucination_like_gpt2"].astype(bool)
    )
    out["hallucination_both_lms"] = (
        out["hallucination_like_qwen"].astype(bool)
        & out["hallucination_like_gpt2"].astype(bool)
    )
    out["lm_hallucination_agreement"] = (
        out["hallucination_like_qwen"].astype(bool)
        == out["hallucination_like_gpt2"].astype(bool)
    )

    out["qwen_gate_status"] = np.where(
        ~out["hallucination_like_qwen"].astype(bool),
        "non_h",
        np.where(out["accepted_best_gate"], "missed_after_gate", "caught_by_gate"),
    )
    out["gpt2_gate_status"] = np.where(
        ~out["hallucination_like_gpt2"].astype(bool),
        "non_h",
        np.where(out["accepted_best_gate"], "missed_after_gate", "caught_by_gate"),
    )

    out["reference_words"] = out["reference"].fillna("").astype(str).str.split().str.len()
    out["hypothesis_words"] = out["hypothesis"].fillna("").astype(str).str.split().str.len()
    out["length_ratio_hyp_ref"] = out["hypothesis_words"] / out["reference_words"].replace(0, np.nan)
    return out


def _summary_row(df: pd.DataFrame, *, lm: str, subset: str) -> Dict[str, object]:
    hall_col = f"hallucination_like_{lm}"
    if subset == "before_all_h":
        block = df[df[hall_col].astype(bool)]
    elif subset == "caught_by_gate":
        block = df[df[hall_col].astype(bool) & df["abstained_best_gate"]]
    elif subset == "after_missed":
        block = df[df[hall_col].astype(bool) & df["accepted_best_gate"]]
    else:
        raise ValueError(subset)

    def mean(col: str) -> float:
        return float(block[col].astype(float).mean()) if len(block) else float("nan")

    def median(col: str) -> float:
        return float(block[col].astype(float).median()) if len(block) else float("nan")

    return {
        "lm": lm,
        "subset": subset,
        "N": int(len(block)),
        "mean_WER": mean("WER"),
        "median_WER": median("WER"),
        "mean_ctc_support_nll": mean("ctc_support_nll"),
        "median_ctc_support_nll": median("ctc_support_nll"),
        "mean_qwen_plaus": mean("qwen_plaus"),
        "mean_gpt2_plaus": mean("gpt2_plaus"),
        "mean_reference_words": mean("reference_words"),
        "mean_hypothesis_words": mean("hypothesis_words"),
        "mean_length_ratio_hyp_ref": mean("length_ratio_hyp_ref"),
        "both_lms_H_rate": float(block["hallucination_both_lms"].mean()) if len(block) else float("nan"),
    }


def _write_lm_subsets(df: pd.DataFrame, outdir: Path, lm: str, columns: List[str]) -> Dict[str, str]:
    hall = df[df[f"hallucination_like_{lm}"].astype(bool)].copy()
    caught = hall[hall["abstained_best_gate"]].copy()
    missed = hall[hall["accepted_best_gate"]].copy()

    paths = {
        "before": outdir / f"{lm}_hallucinations_before.csv",
        "caught": outdir / f"{lm}_hallucinations_caught.csv",
        "after_missed": outdir / f"{lm}_hallucinations_after_missed.csv",
    }
    hall[columns].to_csv(paths["before"], index=False)
    caught[columns].to_csv(paths["caught"], index=False)
    missed[columns].to_csv(paths["after_missed"], index=False)
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate clean hallucinations before/after best acoustic gate")
    parser.add_argument("--scored_outputs", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_clean_dev_coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    args = parser.parse_args()

    df = pd.read_csv(args.scored_outputs)
    _require_columns(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_dev = df[(df["split"] == "dev") & (df["perturbation"] == CLEAN)].copy()
    clean_test = df[(df["split"] == "test") & (df["perturbation"] == CLEAN)].copy()
    if clean_dev.empty or clean_test.empty:
        raise ValueError("Need both clean DEV and clean TEST rows")

    tau, realized_dev_coverage = threshold_for_min_coverage(
        clean_dev["ctc_support_nll"].astype(float), args.target_clean_dev_coverage
    )
    clean_test = _add_analysis_columns(clean_test, tau)

    preferred = [
        "utterance_id",
        "audio_path",
        "reference",
        "hypothesis",
        "WER",
        "qwen_plaus",
        "gpt2_plaus",
        "ctc_support_nll",
        "hallucination_like_qwen",
        "hallucination_like_gpt2",
        "hallucination_both_lms",
        "lm_hallucination_agreement",
        "accepted_best_gate",
        "abstained_best_gate",
        "qwen_gate_status",
        "gpt2_gate_status",
        "reference_words",
        "hypothesis_words",
        "length_ratio_hyp_ref",
    ]
    columns = [c for c in preferred if c in clean_test.columns]

    union = clean_test[clean_test["hallucination_union"]].copy()
    union = union.sort_values(
        ["abstained_best_gate", "ctc_support_nll", "WER"],
        ascending=[False, False, False],
    )
    union_path = args.output_dir / "all_clean_test_hallucinations_union.csv"
    union[columns].to_csv(union_path, index=False)

    lm_paths = {
        lm: _write_lm_subsets(clean_test, args.output_dir, lm, columns)
        for lm in ("qwen", "gpt2")
    }

    summary_rows = []
    for lm in ("qwen", "gpt2"):
        for subset in ("before_all_h", "caught_by_gate", "after_missed"):
            summary_rows.append(_summary_row(clean_test, lm=lm, subset=subset))
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "before_after_hallucination_summary.csv"
    summary.to_csv(summary_path, index=False)

    counts = {
        "clean_test_N": int(len(clean_test)),
        "union_H_N": int(union.shape[0]),
        "qwen_H_before": int(clean_test["hallucination_like_qwen"].astype(bool).sum()),
        "qwen_H_caught": int((clean_test["hallucination_like_qwen"].astype(bool) & clean_test["abstained_best_gate"]).sum()),
        "qwen_H_missed_after": int((clean_test["hallucination_like_qwen"].astype(bool) & clean_test["accepted_best_gate"]).sum()),
        "gpt2_H_before": int(clean_test["hallucination_like_gpt2"].astype(bool).sum()),
        "gpt2_H_caught": int((clean_test["hallucination_like_gpt2"].astype(bool) & clean_test["abstained_best_gate"]).sum()),
        "gpt2_H_missed_after": int((clean_test["hallucination_like_gpt2"].astype(bool) & clean_test["accepted_best_gate"]).sum()),
        "both_lms_H": int(clean_test["hallucination_both_lms"].sum()),
        "test_coverage": float(clean_test["accepted_best_gate"].mean()),
    }

    report = {
        "experiment": "clean_hallucinations_before_after_best_gate",
        "best_gate_definition": "clean DEV 95% target coverage by default; threshold selected from DEV CTC-NLL only",
        "target_clean_dev_coverage": args.target_clean_dev_coverage,
        "realized_clean_dev_coverage": realized_dev_coverage,
        "frozen_threshold": tau,
        "counts": counts,
        "files": {
            "all_hallucinations_union": str(union_path),
            "summary": str(summary_path),
            "qwen": lm_paths["qwen"],
            "gpt2": lm_paths["gpt2"],
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== Clean hallucinations before/after selected gate ===")
    print(f"Target clean DEV coverage: {args.target_clean_dev_coverage:.3f}")
    print(f"Realized clean DEV coverage: {realized_dev_coverage:.4f}")
    print(f"Frozen tau: {tau:.6f}")
    print(f"Clean TEST coverage: {counts['test_coverage']:.4f}")
    print(json.dumps(counts, indent=2))
    print("\nBefore/after summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOutputs:")
    for path in [union_path, summary_path, report_path]:
        print(f"  {path}")
    for lm in ("qwen", "gpt2"):
        for path in lm_paths[lm].values():
            print(f"  {path}")


if __name__ == "__main__":
    main()
