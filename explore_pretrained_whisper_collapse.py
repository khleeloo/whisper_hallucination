#!/usr/bin/env python3
"""Explore exact output collapse in the untouched pretrained Whisper stress run.

Reads the completed pretrained_whisper_stress_pipeline/scored_outputs.csv and
characterizes the dominant normalized hypotheses for each TEST condition.
No model inference is performed.

Outputs
-------
- collapse_condition_summary.csv
- top_hypotheses_by_condition.csv
- top_hypothesis_members.csv
- collapse_report.txt

The analysis is intended to distinguish severe fluent hallucination-like output
from trivial/default cross-utterance collapse under full-noise stress.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from clean_wer_rescore import normalize_asr_text

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_INPUT = ROOT / "pretrained_whisper_stress_pipeline" / "scored_outputs.csv"
DEFAULT_OUTPUT_DIR = ROOT / "pretrained_whisper_stress_pipeline" / "collapse_exploration"
DEFAULT_CONDITIONS = [
    "none",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else float("nan")


def distribution_stats(norm_hyp: pd.Series) -> dict:
    values = norm_hyp.fillna("").astype(str)
    nonempty = values[values != ""]
    n = len(values)
    if n == 0:
        return {}
    counts = nonempty.value_counts()
    probs = counts.to_numpy(dtype=float) / max(len(nonempty), 1)
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    effective_n = float(2.0 ** entropy) if len(probs) else 0.0
    return {
        "N": int(n),
        "N_nonempty": int(len(nonempty)),
        "empty_pct": 100.0 * float((values == "").mean()),
        "N_unique_nonempty": int(counts.size),
        "unique_nonempty_fraction": float(counts.size / len(nonempty)) if len(nonempty) else 0.0,
        "top1_mass": float(counts.iloc[0] / len(nonempty)) if len(nonempty) else 0.0,
        "top5_mass": float(counts.iloc[:5].sum() / len(nonempty)) if len(nonempty) else 0.0,
        "top10_mass": float(counts.iloc[:10].sum() / len(nonempty)) if len(nonempty) else 0.0,
        "output_entropy_bits": entropy,
        "effective_number_outputs": effective_n,
    }


def summarize_top_groups(g: pd.DataFrame, condition: str, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    nonempty = g[g["normalized_hypothesis"] != ""].copy()
    counts = nonempty["normalized_hypothesis"].value_counts().head(top_k)
    summary_rows: List[dict] = []
    member_frames: List[pd.DataFrame] = []

    for rank, (hyp, count) in enumerate(counts.items(), start=1):
        m = nonempty[nonempty["normalized_hypothesis"] == hyp].copy()
        row = {
            "condition": condition,
            "rank": rank,
            "normalized_hypothesis": hyp,
            "count": int(count),
            "mass_all": float(count / len(g)),
            "mass_nonempty": float(count / len(nonempty)) if len(nonempty) else 0.0,
            "WER_mean": safe_mean(m["WER"]),
            "WER_median": float(pd.to_numeric(m["WER"], errors="coerce").median()),
            "qwen_plaus_mean": safe_mean(m["qwen_plaus"]),
            "gpt2_plaus_mean": safe_mean(m["gpt2_plaus"]),
            "ctc_support_nll_mean": safe_mean(m["ctc_support_nll"]) if "ctc_support_nll" in m else float("nan"),
            "strict_qwen_fraction": float(as_bool(m["strict_h_qwen"]).mean()) if "strict_h_qwen" in m else float("nan"),
            "strict_gpt2_fraction": float(as_bool(m["strict_h_gpt2"]).mean()) if "strict_h_gpt2" in m else float("nan"),
            "strict_union_fraction": float(as_bool(m["strict_h_union"]).mean()) if "strict_h_union" in m else float("nan"),
            "rep34_fraction": float(((pd.to_numeric(m.get("rep3", 0), errors="coerce").fillna(0) > 0) | (pd.to_numeric(m.get("rep4", 0), errors="coerce").fillna(0) > 0)).mean()),
            "example_raw_hypothesis": str(m.iloc[0]["hypothesis"]),
            "example_reference": str(m.iloc[0]["reference"]),
        }
        summary_rows.append(row)

        keep = [
            c for c in [
                "split", "utterance_id", "reference", "hypothesis", "WER",
                "qwen_plaus", "gpt2_plaus", "ctc_support_nll",
                "strict_h_qwen", "strict_h_gpt2", "strict_h_union",
            ] if c in m.columns
        ]
        members = m[keep].copy()
        members.insert(0, "rank", rank)
        members.insert(0, "normalized_hypothesis", hyp)
        members.insert(0, "condition", condition)
        member_frames.append(members)

    return pd.DataFrame(summary_rows), pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame()


def render_report(condition_summary: pd.DataFrame, top: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=== Untouched pretrained Whisper: output-collapse exploration ===")
    lines.append("")
    lines.append("Condition-level concentration:")
    cols = [
        "condition", "N", "empty_pct", "N_unique_nonempty",
        "top1_mass", "top5_mass", "top10_mass",
        "output_entropy_bits", "effective_number_outputs",
    ]
    lines.append(condition_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    for condition in condition_summary["condition"]:
        lines.append(f"Top outputs: {condition}")
        sub = top[top["condition"] == condition].head(20)
        show = [
            "rank", "count", "mass_nonempty", "normalized_hypothesis",
            "WER_mean", "qwen_plaus_mean", "gpt2_plaus_mean",
            "strict_qwen_fraction", "strict_gpt2_fraction",
        ]
        lines.append(sub[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")

    lines.append("Interpretation guide:")
    lines.append("- High mass on one/few hypotheses across unrelated references = cross-utterance output collapse.")
    lines.append("- High WER + high LM plausibility within a dominant group = fluent prior-driven hallucination-like collapse.")
    lines.append("- High WER + low LM plausibility within a dominant group = default/degenerate collapse rather than fluent hallucination.")
    lines.append("- Compare 0.50 vs 0.75 to determine whether stronger corruption changes the failure type, not merely its severity.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    required = {
        "split", "perturbation", "reference", "hypothesis", "WER",
        "qwen_plaus", "gpt2_plaus",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    test = df[df["split"].astype(str).str.lower() == "test"].copy()
    test = test[test["perturbation"].isin(args.conditions)].copy()
    test["normalized_hypothesis"] = test["hypothesis"].fillna("").map(normalize_asr_text)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_rows: List[dict] = []
    top_frames: List[pd.DataFrame] = []
    member_frames: List[pd.DataFrame] = []
    for condition in args.conditions:
        g = test[test["perturbation"] == condition].copy()
        if g.empty:
            print(f"WARNING: no TEST rows for {condition}", flush=True)
            continue
        stats = {"condition": condition, **distribution_stats(g["normalized_hypothesis"])}
        stats.update({
            "WER_mean": safe_mean(g["WER"]),
            "qwen_plaus_mean": safe_mean(g["qwen_plaus"]),
            "gpt2_plaus_mean": safe_mean(g["gpt2_plaus"]),
            "strict_qwen_pct": 100.0 * float(as_bool(g["strict_h_qwen"]).mean()) if "strict_h_qwen" in g else float("nan"),
            "strict_gpt2_pct": 100.0 * float(as_bool(g["strict_h_gpt2"]).mean()) if "strict_h_gpt2" in g else float("nan"),
        })
        condition_rows.append(stats)
        top_df, members_df = summarize_top_groups(g, condition, args.top_k)
        top_frames.append(top_df)
        if not members_df.empty:
            member_frames.append(members_df)

    condition_summary = pd.DataFrame(condition_rows)
    top = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    members = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame()

    condition_summary.to_csv(args.output_dir / "collapse_condition_summary.csv", index=False)
    top.to_csv(args.output_dir / "top_hypotheses_by_condition.csv", index=False)
    members.to_csv(args.output_dir / "top_hypothesis_members.csv", index=False)

    report = render_report(condition_summary, top)
    (args.output_dir / "collapse_report.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
