#!/usr/bin/env python3
"""Analyze downloaded MTurk audio-grounding judgments.

The study is stratified by design. This script therefore reports conditional
human-confirmation rates and a severe strict-H vs high-WER non-H contrast; it
does not estimate population hallucination prevalence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of columns found: {list(candidates)}")
    return None


def parse_grounding(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.upper() == "X" or s == "":
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return v if v in {0.0, 1.0, 2.0, 3.0} else np.nan


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def majority_label(values: pd.Series) -> str:
    x = values.dropna().astype(str)
    if x.empty:
        return ""
    vc = x.value_counts()
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        return "tie"
    return str(vc.index[0])


def reshape_results(results: pd.DataFrame) -> pd.DataFrame:
    worker_col = find_col(results, ["WorkerId", "worker_id"])
    assignment_col = find_col(results, ["AssignmentId", "assignment_id"], required=False)
    time_col = find_col(results, ["WorkTimeInSeconds", "work_time_seconds"], required=False)

    rows = []
    for ridx, r in results.iterrows():
        worker = str(r[worker_col])
        assignment = str(r[assignment_col]) if assignment_col else f"row_{ridx}"
        work_time = pd.to_numeric(r[time_col], errors="coerce") if time_col else np.nan
        for pos in range(1, 12):
            k = f"{pos:02d}"
            sid_col = find_col(
                results,
                [f"Answer.sample_id_{k}", f"Input.sample_id_{k}", f"sample_id_{k}"],
                required=False,
            )
            g_col = find_col(results, [f"Answer.grounding_{k}", f"grounding_{k}"], required=False)
            f_col = find_col(results, [f"Answer.failure_type_{k}", f"failure_type_{k}"], required=False)
            if not sid_col or not g_col:
                continue
            sid = str(r[sid_col]).strip() if not pd.isna(r[sid_col]) else ""
            if not sid:
                continue
            raw_g = r[g_col]
            rows.append({
                "worker_id": worker,
                "assignment_id": assignment,
                "work_time_seconds": work_time,
                "position": pos,
                "sample_id": sid,
                "grounding_raw": raw_g,
                "grounding": parse_grounding(raw_g),
                "cannot_judge": str(raw_g).strip().upper() == "X",
                "failure_type": str(r[f_col]).strip() if f_col and not pd.isna(r[f_col]) else "",
            })
    if not rows:
        raise ValueError("No MTurk item judgments could be parsed. Check exported column names.")
    return pd.DataFrame(rows)


def worker_qc(long: pd.DataFrame, qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = qc[["sample_id", "expected_grounding", "expected_class"]].copy()
    z = long.merge(key, on="sample_id", how="inner")
    z["qc_pass"] = False
    pos = pd.to_numeric(z["expected_grounding"], errors="coerce") == 3
    neg = pd.to_numeric(z["expected_grounding"], errors="coerce") == 0
    z.loc[pos, "qc_pass"] = pd.to_numeric(z.loc[pos, "grounding"], errors="coerce") >= 2
    z.loc[neg, "qc_pass"] = pd.to_numeric(z.loc[neg, "grounding"], errors="coerce") <= 1

    rows = []
    for worker, g in z.groupby("worker_id"):
        total = len(g)
        passed = int(g["qc_pass"].sum())
        failed = total - passed
        acc = passed / total if total else np.nan
        exclude = (total >= 2 and failed >= 2) or (total >= 4 and acc < 0.75)
        rows.append({"worker_id": worker, "qc_total": total, "qc_pass": passed,
                     "qc_fail": failed, "qc_accuracy": acc, "exclude_qc": exclude})
    return z, pd.DataFrame(rows)


def aggregate_samples(exp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in exp.groupby("sample_id"):
        valid = pd.to_numeric(g["grounding"], errors="coerce").dropna()
        unsupported = int((valid <= 1).sum())
        supported = int((valid >= 2).sum())
        n = len(valid)
        if n == 0:
            maj = np.nan
        elif unsupported > n / 2:
            maj = True
        elif supported > n / 2:
            maj = False
        else:
            maj = np.nan
        rows.append({
            "sample_id": sid,
            "n_valid_ratings": n,
            "n_cannot_judge": int(g["cannot_judge"].sum()),
            "grounding_mean": float(valid.mean()) if n else np.nan,
            "grounding_median": float(valid.median()) if n else np.nan,
            "unsupported_votes": unsupported,
            "supported_votes": supported,
            "majority_unsupported": maj,
            "majority_failure_type": majority_label(g["failure_type"]),
        })
    return pd.DataFrame(rows)


def summarize_groups(samples: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for name, g in samples.groupby(group_col, dropna=False):
        usable = g[g["majority_unsupported"].notna()].copy()
        k = int(usable["majority_unsupported"].astype(bool).sum())
        n = len(usable)
        lo, hi = wilson(k, n)
        rows.append({
            group_col: name,
            "N_samples": len(g),
            "N_with_majority": n,
            "human_unsupported_N": k,
            "human_unsupported_rate": k / n if n else np.nan,
            "human_unsupported_ci95_low": lo,
            "human_unsupported_ci95_high": hi,
            "mean_grounding": float(g["grounding_mean"].mean()),
            "cannot_judge_total": int(g["n_cannot_judge"].sum()),
        })
    return pd.DataFrame(rows)


def primary_contrast(samples: pd.DataFrame) -> pd.DataFrame:
    severe_h = samples[samples["stratum"].isin([
        "raw_severe_strict_Hq", "adapted_severe_strict_Hq", "seamless_severe_strict_Hq"
    ])]
    control = samples[samples["stratum"] == "high_WER_non_H_control"]

    def stats(g):
        x = g[g["majority_unsupported"].notna()]
        n = len(x); k = int(x["majority_unsupported"].astype(bool).sum())
        lo, hi = wilson(k, n)
        return n, k, (k / n if n else np.nan), lo, hi

    n1, k1, p1, lo1, hi1 = stats(severe_h)
    n0, k0, p0, lo0, hi0 = stats(control)
    rr = p1 / p0 if p0 and np.isfinite(p0) else np.nan
    rd = p1 - p0 if np.isfinite(p1) and np.isfinite(p0) else np.nan
    return pd.DataFrame([
        {"group": "severe_strict_Hq", "N": n1, "unsupported_N": k1,
         "unsupported_rate": p1, "ci95_low": lo1, "ci95_high": hi1,
         "risk_difference_vs_control": rd, "risk_ratio_vs_control": rr},
        {"group": "high_WER_non_H_control", "N": n0, "unsupported_N": k0,
         "unsupported_rate": p0, "ci95_low": lo0, "ci95_high": hi0,
         "risk_difference_vs_control": 0.0, "risk_ratio_vs_control": 1.0},
    ])


def agreement_alpha(exp: pd.DataFrame) -> Dict[str, object]:
    try:
        import krippendorff
    except ImportError:
        return {"ordinal_krippendorff_alpha": None,
                "note": "Install `krippendorff` to compute ordinal alpha."}
    z = exp.dropna(subset=["grounding"]).copy()
    if z.empty:
        return {"ordinal_krippendorff_alpha": None, "note": "No valid ratings"}
    mat = z.pivot_table(index="worker_id", columns="sample_id", values="grounding", aggfunc="first")
    alpha = krippendorff.alpha(reliability_data=mat.to_numpy(dtype=float), level_of_measurement="ordinal")
    return {"ordinal_krippendorff_alpha": float(alpha),
            "n_workers": int(mat.shape[0]), "n_samples": int(mat.shape[1])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--private_manifest", type=Path, required=True)
    p.add_argument("--qc_key", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--fast_seconds", type=float, default=90.0,
                   help="Flag assignments faster than this; not excluded unless --exclude_fast is set.")
    p.add_argument("--exclude_fast", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results)
    private = pd.read_csv(args.private_manifest)
    qc = pd.read_csv(args.qc_key)
    long = reshape_results(results)
    qc_long, workers = worker_qc(long, qc)

    all_workers = pd.DataFrame({"worker_id": long["worker_id"].unique()})
    workers = all_workers.merge(workers, on="worker_id", how="left")
    for c, default in [("qc_total", 0), ("qc_pass", 0), ("qc_fail", 0), ("exclude_qc", False)]:
        workers[c] = workers[c].fillna(default)
    workers["fast_assignment_any"] = workers["worker_id"].map(
        long.assign(fast=long["work_time_seconds"] < args.fast_seconds)
            .groupby("worker_id")["fast"].any()
    ).fillna(False)
    workers["excluded"] = workers["exclude_qc"].astype(bool)
    if args.exclude_fast:
        workers["excluded"] |= workers["fast_assignment_any"].astype(bool)

    long = long.merge(workers[["worker_id", "excluded"]], on="worker_id", how="left")
    long["excluded"] = long["excluded"].fillna(False)
    qc_ids = set(qc["sample_id"].astype(str))
    exp = long[(~long["sample_id"].isin(qc_ids)) & (~long["excluded"].astype(bool))].copy()
    exp = exp.merge(private, on="sample_id", how="left", validate="many_to_one")

    samples = aggregate_samples(exp)
    samples = samples.merge(private, on="sample_id", how="left", validate="one_to_one")

    long.to_csv(args.output_dir / "all_judgments_long.csv", index=False)
    qc_long.to_csv(args.output_dir / "qc_judgments.csv", index=False)
    workers.to_csv(args.output_dir / "worker_qc_summary.csv", index=False)
    exp.to_csv(args.output_dir / "experimental_judgments_included.csv", index=False)
    samples.to_csv(args.output_dir / "sample_level_majority.csv", index=False)
    summarize_groups(samples, "stratum").to_csv(args.output_dir / "summary_by_stratum.csv", index=False)
    summarize_groups(samples, "model").to_csv(args.output_dir / "summary_by_model.csv", index=False)
    summarize_groups(samples, "condition").to_csv(args.output_dir / "summary_by_condition.csv", index=False)
    primary_contrast(samples).to_csv(args.output_dir / "primary_contrast.csv", index=False)

    agreement = agreement_alpha(exp)
    agreement.update({
        "fast_seconds_flag": args.fast_seconds,
        "exclude_fast": args.exclude_fast,
        "n_workers_total": int(len(workers)),
        "n_workers_excluded": int(workers["excluded"].sum()),
        "n_experimental_judgments_included": int(len(exp)),
        "n_samples_with_any_included_judgment": int(samples["sample_id"].nunique()),
    })
    (args.output_dir / "agreement_and_qc.json").write_text(
        json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== Primary contrast ===")
    print(primary_contrast(samples).to_string(index=False))
    print("\n=== QC/agreement ===")
    print(json.dumps(agreement, indent=2, sort_keys=True))
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
