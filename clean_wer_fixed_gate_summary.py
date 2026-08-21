#!/usr/bin/env python3
"""Re-evaluate corrected hallucination labels under the original frozen CTC gate.

This keeps the mitigation intervention fixed while changing only WER normalization,
so before/after differences are attributable to corrected evaluation rather than
to a newly selected gate threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from clean_wer_rescore import summarize_gate

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_RESCORED = ROOT / "clean_wer_rescore" / "scored_outputs_cleanwer.csv"
DEFAULT_GATE = ROOT / "hallucination_mitigation_acoustic_before_after" / "frozen_gate_threshold.json"
DEFAULT_OUTPUT = ROOT / "clean_wer_rescore" / "test_mitigation_summary_cleanwer.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Corrected-WER TEST summary under original frozen gate")
    p.add_argument("--rescored", type=Path, default=DEFAULT_RESCORED)
    p.add_argument("--gate_json", type=Path, default=DEFAULT_GATE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    df = pd.read_csv(args.rescored)
    payload = json.loads(args.gate_json.read_text(encoding="utf-8"))
    tau = float(payload["threshold"])

    test = df[df["split"].astype(str) == "test"].copy()
    rows = []
    for _, group in test.groupby("perturbation", sort=False):
        row = summarize_gate(group, tau)
        row["threshold"] = tau
        row["gate_source"] = str(args.gate_json)
        row["gate_status"] = "original_DEV_tuned_gate_frozen_before_clean_WER_rescore"
        rows.append(row)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("=== CORRECTED WER WITH ORIGINAL FROZEN GATE ===")
    print(f"tau={tau:.6f}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
