#!/usr/bin/env python3
"""Test whether acoustic-consistency abstention transfers to ordinary clean ASR.

This is the second paper-facing mitigation experiment.  It does *not* rerun
Whisper, Qwen3, GPT-2, or wav2vec2.  Instead it reuses the DEV/TEST outputs and
normalized wav2vec2-CTC hypothesis NLLs produced by
``acoustic_abstention_before_after.py``.

Question
--------
The first mitigation experiment showed that a DEV-tuned CTC gate nearly
perfectly rejects the severe full-noise fallback regime while preserving about
98% clean coverage.  That alone does not show transfer to hallucination-like
outputs from the original clean model.  Here we test whether the *same acoustic
score* preferentially identifies clean hallucination-like outputs when the
operating point is relaxed.

Protocol
--------
1. Reuse the frozen hallucination labels and CTC scores from the completed
   before/after experiment.
2. On clean DEV only, choose CTC thresholds that achieve target clean coverage
   levels (99%, 98%, 95%, 90%, 85%).  Threshold selection uses only the CTC
   score distribution and the requested coverage; TEST labels are never used.
3. Freeze each threshold and evaluate it on clean TEST and both severe-noise
   TEST conditions.
4. Report, separately for Qwen3 and GPT-2:
      - clean hallucination capture recall,
      - hallucination rate among emitted clean transcripts,
      - precision of clean rejections for hallucination-like outputs,
      - system-level emitted hallucination incidence,
      - accepted-output WER,
      - severe-noise coverage and hallucination capture using the same tau.
5. Also report clean DEV/TEST ROC-AUC and average precision for CTC NLL as a
   *descriptive separability analysis*.  Higher CTC NLL means weaker acoustic
   support and therefore higher predicted risk.

The original stress-tuned threshold is included as an additional operating
point when ``frozen_gate_threshold.json`` is available.  It is not retuned.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from acoustic_abstention_mitigation import accepted_mask


ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_SOURCE_DIR = ROOT / "hallucination_mitigation_acoustic_before_after"
DEFAULT_SCORED = DEFAULT_SOURCE_DIR / "scored_outputs.csv"
DEFAULT_GATE_JSON = DEFAULT_SOURCE_DIR / "frozen_gate_threshold.json"
DEFAULT_OUTPUT_DIR = ROOT / "hallucination_mitigation_acoustic_clean_transfer"
DEFAULT_COVERAGES = [0.99, 0.98, 0.95, 0.90, 0.85]
CLEAN = "none"
FULL_05 = "full_noise_amp0.5_dur0.0"
FULL_075 = "full_noise_amp0.75_dur0.0"
CONDITIONS = [CLEAN, FULL_05, FULL_075]


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "split",
        "perturbation",
        "WER",
        "ctc_support_nll",
        "hallucination_like_qwen",
        "hallucination_like_gpt2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"scored_outputs.csv is missing required columns: {missing}")
    needed_groups = {(s, c) for s in ("dev", "test") for c in CONDITIONS}
    present = set(zip(df["split"].astype(str), df["perturbation"].astype(str)))
    missing_groups = sorted(needed_groups - present)
    if missing_groups:
        raise ValueError(f"Missing split/condition groups: {missing_groups}")


def threshold_for_min_coverage(scores: Sequence[float], target: float) -> Tuple[float, float]:
    """Smallest threshold whose empirical coverage is >= target.

    The gate accepts iff finite CTC NLL <= tau.  Selecting the smallest feasible
    tau is the strongest abstention rule compatible with the requested clean
    coverage.  Ties can make realized coverage slightly larger than target.
    """
    if not (0.0 < target <= 1.0):
        raise ValueError("target coverage must be in (0, 1]")
    values = np.asarray(scores, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("cannot calibrate coverage on an empty set")
    finite = np.sort(values[np.isfinite(values)])
    required = int(math.ceil(target * n - 1e-12))
    if len(finite) < required:
        max_cov = len(finite) / n
        raise ValueError(
            f"Target coverage {target:.3f} is impossible because finite-score coverage is only {max_cov:.3f}"
        )
    tau = float(finite[required - 1])
    realized = float(accepted_mask(values, tau).mean())
    return tau, realized


def binary_ranking_metrics(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, float]:
    """Tie-aware ROC-AUC and average precision with larger score = more positive."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    p = int(labels.sum())
    n = int((~labels).sum())
    if p == 0 or n == 0:
        return {"roc_auc": float("nan"), "average_precision": float("nan"), "N": int(len(labels)), "positives": p}

    # Aggregate ties before building the curve so ranking metrics are not
    # sensitive to arbitrary row order within equal CTC scores.
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    boundaries = np.r_[np.flatnonzero(np.diff(s) != 0) + 1, len(s)]
    starts = np.r_[0, boundaries[:-1]]
    pos_by_group = np.array([int(y[a:b].sum()) for a, b in zip(starts, boundaries)], dtype=float)
    size_by_group = (boundaries - starts).astype(float)
    neg_by_group = size_by_group - pos_by_group

    tp = np.cumsum(pos_by_group)
    fp = np.cumsum(neg_by_group)
    tpr = np.r_[0.0, tp / p]
    fpr = np.r_[0.0, fp / n]
    roc_auc = float(np.trapezoid(tpr, fpr))

    recall = tp / p
    precision = tp / (tp + fp)
    recall_prev = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - recall_prev) * precision))
    return {"roc_auc": roc_auc, "average_precision": average_precision, "N": int(len(labels)), "positives": p}


def summarize_condition(group: pd.DataFrame, tau: float, lm: str) -> Dict[str, float]:
    hall_col = f"hallucination_like_{lm}"
    hall = group[hall_col].astype(bool).to_numpy()
    accept = accepted_mask(group["ctc_support_nll"].astype(float).to_numpy(), tau)
    reject = ~accept
    emitted_h = hall & accept
    rejected_h = hall & reject
    n_accept = int(accept.sum())
    n_reject = int(reject.sum())
    n_hall = int(hall.sum())
    wer = group["WER"].astype(float).to_numpy()
    before = float(hall.mean())
    among_emitted = float(emitted_h.sum() / n_accept) if n_accept else float("nan")
    return {
        "coverage": float(accept.mean()),
        "abstention_rate": float(reject.mean()),
        "hallucination_rate_before": before,
        "system_emitted_hallucination_incidence": float(emitted_h.mean()),
        "hallucination_rate_among_emitted": among_emitted,
        "hallucination_capture_recall": float(rejected_h.sum() / n_hall) if n_hall else 0.0,
        "rejection_precision_for_hallucination": float(rejected_h.sum() / n_reject) if n_reject else float("nan"),
        "relative_risk_reduction_among_emitted": (
            float(1.0 - among_emitted / before) if before > 0 and math.isfinite(among_emitted) else float("nan")
        ),
        "mean_WER_all": float(np.mean(wer)),
        "mean_WER_accepted": float(np.mean(wer[accept])) if n_accept else float("nan"),
        "N": int(len(group)),
        "N_accepted": n_accept,
        "N_hallucination": n_hall,
    }


def evaluate_operating_point(df: pd.DataFrame, name: str, tau: float, target: float | None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    test = df[df["split"] == "test"]
    for condition in CONDITIONS:
        group = test[test["perturbation"] == condition]
        for lm in ("qwen", "gpt2"):
            metrics = summarize_condition(group, tau, lm)
            rows.append(
                {
                    "operating_point": name,
                    "target_clean_dev_coverage": target,
                    "threshold": tau,
                    "split": "test",
                    "condition": condition,
                    "lm": lm,
                    **metrics,
                }
            )
    return rows


def paper_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for op, block in long_df.groupby("operating_point", sort=False):
        first = block.iloc[0]
        row: Dict[str, object] = {
            "operating_point": op,
            "target_clean_dev_coverage": first["target_clean_dev_coverage"],
            "threshold": first["threshold"],
        }
        for lm in ("qwen", "gpt2"):
            clean = block[(block["condition"] == CLEAN) & (block["lm"] == lm)].iloc[0]
            row[f"clean_coverage_{lm}"] = clean["coverage"]
            row[f"clean_H_before_{lm}"] = clean["hallucination_rate_before"]
            row[f"clean_H_among_emitted_{lm}"] = clean["hallucination_rate_among_emitted"]
            row[f"clean_H_capture_{lm}"] = clean["hallucination_capture_recall"]
            row[f"clean_rejection_precision_{lm}"] = clean["rejection_precision_for_hallucination"]
            row[f"clean_relative_H_risk_reduction_{lm}"] = clean["relative_risk_reduction_among_emitted"]
            row[f"clean_WER_emitted_{lm}"] = clean["mean_WER_accepted"]
            for condition, suffix in ((FULL_05, "full05"), (FULL_075, "full075")):
                stress = block[(block["condition"] == condition) & (block["lm"] == lm)].iloc[0]
                row[f"{suffix}_coverage_{lm}"] = stress["coverage"]
                row[f"{suffix}_H_capture_{lm}"] = stress["hallucination_capture_recall"]
                row[f"{suffix}_H_among_emitted_{lm}"] = stress["hallucination_rate_among_emitted"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-transfer risk/coverage analysis for wav2vec2 acoustic abstention")
    parser.add_argument("--scored_outputs", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--frozen_gate_json", type=Path, default=DEFAULT_GATE_JSON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    args = parser.parse_args()

    df = pd.read_csv(args.scored_outputs)
    validate_input(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_dev = df[(df["split"] == "dev") & (df["perturbation"] == CLEAN)].copy()
    operating_points: List[Tuple[str, float, float | None]] = []

    if args.frozen_gate_json.exists():
        payload = json.loads(args.frozen_gate_json.read_text(encoding="utf-8"))
        tau = float(payload["threshold"])
        operating_points.append(("stress_tuned_original", tau, None))

    for coverage in args.coverages:
        tau, realized = threshold_for_min_coverage(clean_dev["ctc_support_nll"], coverage)
        print(
            f"DEV target clean coverage={coverage:.3f}: tau={tau:.6f}, realized={realized:.4f}",
            flush=True,
        )
        operating_points.append((f"clean_cov_{coverage:.3f}", tau, coverage))

    long_rows: List[Dict[str, object]] = []
    for name, tau, target in operating_points:
        long_rows.extend(evaluate_operating_point(df, name, tau, target))
    long_df = pd.DataFrame(long_rows)
    long_path = args.output_dir / "clean_transfer_by_condition.csv"
    long_df.to_csv(long_path, index=False)

    wide = paper_wide_table(long_df)
    wide_path = args.output_dir / "paper_clean_transfer_summary.csv"
    wide.to_csv(wide_path, index=False)

    separability: Dict[str, object] = {}
    for split in ("dev", "test"):
        clean = df[(df["split"] == split) & (df["perturbation"] == CLEAN)]
        separability[split] = {}
        for lm in ("qwen", "gpt2"):
            separability[split][lm] = binary_ranking_metrics(
                clean["ctc_support_nll"], clean[f"hallucination_like_{lm}"].astype(bool)
            )
    sep_path = args.output_dir / "clean_transfer_separability.json"
    sep_path.write_text(json.dumps(separability, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "experiment": "acoustic_abstention_clean_transfer_risk_coverage",
        "question": (
            "Does the wav2vec2 acoustic-consistency score that rejects the magnified full-noise prior-fallback regime "
            "also preferentially identify hallucination-like outputs from the original clean Whisper model?"
        ),
        "threshold_selection": (
            "Each coverage-sweep threshold is selected from clean DEV CTC NLL only as the smallest tau achieving the "
            "requested empirical clean coverage; thresholds are frozen before TEST evaluation."
        ),
        "coverages": args.coverages,
        "source_scored_outputs": str(args.scored_outputs),
        "source_frozen_gate_json": str(args.frozen_gate_json),
        "outputs": {
            "long_by_condition": str(long_path),
            "paper_summary": str(wide_path),
            "clean_separability": str(sep_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== Clean TEST transfer summary ===", flush=True)
    cols = [
        "operating_point",
        "threshold",
        "clean_coverage_qwen",
        "clean_H_before_qwen",
        "clean_H_among_emitted_qwen",
        "clean_H_capture_qwen",
        "clean_rejection_precision_qwen",
        "full05_H_capture_qwen",
        "full075_H_capture_qwen",
    ]
    print(wide[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\nClean CTC-NLL separability:", json.dumps(separability, indent=2), flush=True)
    print("\nOutputs:")
    for p in (wide_path, long_path, sep_path, report_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()
