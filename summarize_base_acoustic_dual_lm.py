#!/usr/bin/env python3
"""Summarize the Base acoustic-stress experiment with Qwen3 and GPT2 labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        boot[i] = values[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.paired_csv)
    base = df[df["condition"] == "Base"].copy()
    if base.empty:
        raise ValueError("No Base rows in paired CSV")

    required = {
        "WER", "WER_clean",
        "PlausibilityQwen", "PlausibilityQwen_clean",
        "PlausibilityGPT2", "PlausibilityGPT2_clean",
        "HallucinationQwen", "HallucinationQwen_clean", "delta_HallucinationQwen",
        "HallucinationGPT2", "HallucinationGPT2_clean", "delta_HallucinationGPT2",
    }
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Paired CSV is not dual-LM complete; missing {sorted(missing)}")

    rows = []
    for ordinal, (perturbation, g) in enumerate(base.groupby("perturbation", sort=True)):
        q_delta = pd.to_numeric(g["delta_HallucinationQwen"], errors="coerce").to_numpy(float)
        g_delta = pd.to_numeric(g["delta_HallucinationGPT2"], errors="coerce").to_numpy(float)
        q_lo, q_hi = bootstrap_mean_ci(q_delta, args.bootstrap, args.seed + ordinal * 2)
        g_lo, g_hi = bootstrap_mean_ci(g_delta, args.bootstrap, args.seed + ordinal * 2 + 1)
        rows.append({
            "perturbation": perturbation,
            "N": int(len(g)),
            "clean_WER": float(g["WER_clean"].mean()),
            "stressed_WER": float(g["WER"].mean()),
            "delta_WER": float((g["WER"] - g["WER_clean"]).mean()),
            "clean_qwen_plaus": float(g["PlausibilityQwen_clean"].mean()),
            "stressed_qwen_plaus": float(g["PlausibilityQwen"].mean()),
            "clean_gpt2_plaus": float(g["PlausibilityGPT2_clean"].mean()),
            "stressed_gpt2_plaus": float(g["PlausibilityGPT2"].mean()),
            "clean_hall_qwen": float(g["HallucinationQwen_clean"].mean()),
            "stressed_hall_qwen": float(g["HallucinationQwen"].mean()),
            "delta_hall_qwen": float(np.nanmean(q_delta)),
            "qwen_delta_CI_low": q_lo,
            "qwen_delta_CI_high": q_hi,
            "qwen_significant_increase": bool(q_lo > 0),
            "clean_hall_gpt2": float(g["HallucinationGPT2_clean"].mean()),
            "stressed_hall_gpt2": float(g["HallucinationGPT2"].mean()),
            "delta_hall_gpt2": float(np.nanmean(g_delta)),
            "gpt2_delta_CI_low": g_lo,
            "gpt2_delta_CI_high": g_hi,
            "gpt2_significant_increase": bool(g_lo > 0),
        })

    out = pd.DataFrame(rows).sort_values("delta_hall_qwen", ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print("=== Base acoustic response: Qwen3 primary + GPT2 parallel ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
