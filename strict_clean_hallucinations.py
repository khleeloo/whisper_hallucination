#!/usr/bin/env python3
"""Extract paper-facing strict hallucination-like candidates from corrected clean WER scores.

Uses the cached/corrected outputs produced by ``clean_wer_rescore.py``. No model
inference or rescoring is performed here.

Strict label (separately for each LM):
    H_strict_Qwen = valid reference AND WER > 0.50 AND Qwen plaus > frozen clean-DEV Qwen threshold
    H_strict_GPT2 = valid reference AND WER > 0.50 AND GPT2 plaus > frozen clean-DEV GPT2 threshold

The WER cutoff is configurable, but defaults to 0.50 to isolate severe fluent
recognition failures rather than ordinary above-average ASR errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_INPUT = ROOT / "clean_wer_rescore" / "scored_outputs_cleanwer.csv"
DEFAULT_THRESHOLDS = ROOT / "clean_wer_rescore" / "frozen_hallucination_thresholds_cleanwer.json"
DEFAULT_OUTPUT = ROOT / "clean_wer_rescore" / "strict_hallucinations"
DEFAULT_GATE_TAU = 1.468906  # clean-DEV 95% coverage operating point


def main() -> None:
    p = argparse.ArgumentParser(description="Extract strict clean hallucination-like candidates")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--wer_threshold", type=float, default=0.50)
    p.add_argument("--gate_tau", type=float, default=DEFAULT_GATE_TAU)
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.thresholds.exists():
        raise FileNotFoundError(args.thresholds)

    df = pd.read_csv(args.input)
    th = json.loads(args.thresholds.read_text(encoding="utf-8"))
    required = {
        "split", "perturbation", "utterance_id", "audio_path", "reference", "hypothesis",
        "WER", "qwen_plaus", "gpt2_plaus", "ctc_support_nll", "valid_reference_cleanwer",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = df[(df["split"].astype(str) == "test") & (df["perturbation"].astype(str) == "none")].copy()
    valid = clean["valid_reference_cleanwer"].astype(bool) & np.isfinite(pd.to_numeric(clean["WER"], errors="coerce"))
    high_wer = valid & (pd.to_numeric(clean["WER"], errors="coerce") > float(args.wer_threshold))

    qwen_thr = float(th["qwen_plausibility_threshold"])
    gpt2_thr = float(th["gpt2_plausibility_threshold"])
    clean["strict_h_qwen"] = high_wer & (pd.to_numeric(clean["qwen_plaus"], errors="coerce") > qwen_thr)
    clean["strict_h_gpt2"] = high_wer & (pd.to_numeric(clean["gpt2_plaus"], errors="coerce") > gpt2_thr)
    clean["strict_h_both"] = clean["strict_h_qwen"] & clean["strict_h_gpt2"]
    clean["strict_h_union"] = clean["strict_h_qwen"] | clean["strict_h_gpt2"]
    clean["accepted_gate95"] = np.isfinite(pd.to_numeric(clean["ctc_support_nll"], errors="coerce")) & (
        pd.to_numeric(clean["ctc_support_nll"], errors="coerce") <= float(args.gate_tau)
    )
    clean["caught_gate95"] = clean["strict_h_union"] & ~clean["accepted_gate95"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "utterance_id", "audio_path", "reference", "hypothesis", "WER",
        "qwen_plaus", "gpt2_plaus", "ctc_support_nll",
        "strict_h_qwen", "strict_h_gpt2", "strict_h_both", "strict_h_union",
        "accepted_gate95", "caught_gate95",
    ]
    for optional in [
        "reference_norm_cleanwer", "hypothesis_norm_cleanwer",
        "reference_words_cleanwer", "hypothesis_words_cleanwer",
    ]:
        if optional in clean.columns:
            cols.append(optional)

    strict = clean[clean["strict_h_union"]].copy().sort_values(
        ["strict_h_both", "WER", "ctc_support_nll"], ascending=[False, False, False]
    )
    strict[cols].to_csv(args.output_dir / "strict_clean_hallucinations_union.csv", index=False)
    strict[strict["strict_h_both"]][cols].to_csv(args.output_dir / "strict_clean_hallucinations_both_lms.csv", index=False)
    strict[strict["strict_h_qwen"]][cols].to_csv(args.output_dir / "strict_clean_hallucinations_qwen.csv", index=False)
    strict[strict["strict_h_gpt2"]][cols].to_csv(args.output_dir / "strict_clean_hallucinations_gpt2.csv", index=False)
    strict[strict["caught_gate95"]][cols].to_csv(args.output_dir / "strict_clean_hallucinations_caught_gate95.csv", index=False)
    strict[~strict["caught_gate95"]][cols].to_csv(args.output_dir / "strict_clean_hallucinations_missed_gate95.csv", index=False)

    summary = {
        "N_clean_test": int(len(clean)),
        "N_valid_reference": int(valid.sum()),
        "strict_wer_threshold": float(args.wer_threshold),
        "criterion": "valid reference AND WER > strict threshold AND LM plausibility > frozen clean-DEV LM mean",
        "qwen_plausibility_threshold": qwen_thr,
        "gpt2_plausibility_threshold": gpt2_thr,
        "N_qwen": int(clean["strict_h_qwen"].sum()),
        "N_gpt2": int(clean["strict_h_gpt2"].sum()),
        "N_both": int(clean["strict_h_both"].sum()),
        "N_union": int(clean["strict_h_union"].sum()),
        "qwen_rate_pct": 100.0 * float(clean["strict_h_qwen"].mean()),
        "gpt2_rate_pct": 100.0 * float(clean["strict_h_gpt2"].mean()),
        "both_rate_pct": 100.0 * float(clean["strict_h_both"].mean()),
        "union_rate_pct": 100.0 * float(clean["strict_h_union"].mean()),
        "gate95_tau": float(args.gate_tau),
        "N_union_caught_gate95": int(clean["caught_gate95"].sum()),
        "union_gate95_capture_pct": (
            100.0 * float(clean["caught_gate95"].sum() / clean["strict_h_union"].sum())
            if int(clean["strict_h_union"].sum()) else 0.0
        ),
    }
    (args.output_dir / "strict_clean_hallucinations_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== Strict clean hallucination-like candidates ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\n=== Candidates (union; both-LM first, then WER descending) ===")
    if strict.empty:
        print("None")
    else:
        display = strict[["utterance_id", "reference", "hypothesis", "WER", "qwen_plaus", "gpt2_plaus", "ctc_support_nll", "strict_h_both", "caught_gate95"]]
        print(display.to_string(index=False, max_colwidth=100, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
