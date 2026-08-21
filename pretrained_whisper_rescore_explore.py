#!/usr/bin/env python3
"""Rescore and explore the untouched pretrained Whisper stress experiment.

This script is intentionally inference-free. It reuses the existing per-output
Whisper/Qwen/GPT-2/wav2vec2 scores, fixes a WER-normalization bug in which the old
regex could turn ordinary words ending in ``nt`` into ``... not`` (e.g. want ->
wa not, instrument -> instrume not), recomputes all WER-dependent labels and DEV
thresholds, recalibrates the frozen acoustic-support gate, and explores dominant
outputs under acoustic stress.

Inputs
------
  /scratch/vemotionsys/rmfrieske/whisper_hallucination/
      pretrained_whisper_stress_pipeline/scored_outputs.csv

Outputs are written to ``pretrained_whisper_stress_pipeline/rescore_explore``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from acoustic_abstention_mitigation import accepted_mask, select_gate_threshold
from pretrained_whisper_stress_pipeline import (
    apply_labels,
    derive_thresholds,
    gate_test_summary,
    summarize_conditions,
)

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_SOURCE = ROOT / "pretrained_whisper_stress_pipeline" / "scored_outputs.csv"
DEFAULT_OUTPUT_DIR = ROOT / "pretrained_whisper_stress_pipeline" / "rescore_explore"
INVALID_REFERENCES = {"", "undefined", "none", "null", "nan", "n/a", "na"}
WHISPER_SPECIAL = re.compile(r"<\|[^|]+\|>")


def normalize_asr_text_v2(text: object) -> str:
    """Conservative token-level WER normalization.

    Important difference from the previous implementation: ``n't`` expansion
    requires an apostrophe. The previous ``n['’]?t`` pattern made the apostrophe
    optional and therefore corrupted normal lexical items ending in ``nt``.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = WHISPER_SPECIAL.sub(" ", s)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")

    # Expand only contractions that are explicitly marked as contractions.
    s = re.sub(r"\bwon't\b", "will not", s)
    s = re.sub(r"\bcan't\b", "can not", s)
    s = re.sub(r"\bshan't\b", "shall not", s)
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r"'ll\b", " will", s)
    s = re.sub(r"'re\b", " are", s)
    s = re.sub(r"'ve\b", " have", s)
    s = re.sub(r"'m\b", " am", s)

    # Treat typographic token boundaries as spaces rather than deleting them.
    s = re.sub(r"[-‐‑‒–—―/]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize decoder outputs that already contain split contraction pieces.
    s = re.sub(r"\b(i|you|he|she|it|we|they)\s+ll\b", r"\1 will", s)
    s = re.sub(r"\b(you|we|they)\s+re\b", r"\1 are", s)
    s = re.sub(r"\b(i|you|we|they)\s+ve\b", r"\1 have", s)
    s = re.sub(r"\bi\s+m\b", "i am", s)
    s = re.sub(r"\b(can)\s+n\s+t\b", r"\1 not", s)
    s = re.sub(r"\b(will)\s+n\s+t\b", r"\1 not", s)
    s = re.sub(r"\b(\w+)\s+n\s+t\b", r"\1 not", s)
    return re.sub(r"\s+", " ", s).strip()


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def clean_wer_v2(reference: object, hypothesis: object) -> Tuple[float, str, str, bool]:
    ref = normalize_asr_text_v2(reference)
    hyp = normalize_asr_text_v2(hypothesis)
    valid = ref not in INVALID_REFERENCES
    if not valid:
        return float("nan"), ref, hyp, False
    rw, hw = ref.split(), hyp.split()
    return float(_levenshtein(rw, hw) / len(rw)), ref, hyp, True


def add_clean_wer_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["WER_previous"] = pd.to_numeric(out.get("WER"), errors="coerce")
    out["reference_norm_previous"] = out.get("reference_norm_cleanwer", "")
    out["hypothesis_norm_previous"] = out.get("hypothesis_norm_cleanwer", "")
    rows = [clean_wer_v2(r, h) for r, h in zip(out["reference"], out["hypothesis"])]
    out["WER"] = [x[0] for x in rows]
    out["reference_norm_cleanwer"] = [x[1] for x in rows]
    out["hypothesis_norm_cleanwer"] = [x[2] for x in rows]
    out["valid_reference_cleanwer"] = [x[3] for x in rows]
    out["reference_words_cleanwer"] = out["reference_norm_cleanwer"].map(lambda s: len(str(s).split()))
    out["hypothesis_words_cleanwer"] = out["hypothesis_norm_cleanwer"].map(lambda s: len(str(s).split()))
    out["WER_changed"] = ~np.isclose(
        pd.to_numeric(out["WER_previous"], errors="coerce"),
        pd.to_numeric(out["WER"], errors="coerce"),
        equal_nan=True,
    )
    return out


def save_top_hypotheses(test: pd.DataFrame, tau: float, outdir: Path, top_k: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for condition, g in test.groupby("perturbation", sort=False):
        work = g.copy()
        work["hyp_norm_v2"] = work["hypothesis"].map(normalize_asr_text_v2)
        work["gate_accepted"] = accepted_mask(work["ctc_support_nll"].astype(float), tau)
        counts = work["hyp_norm_v2"].value_counts(dropna=False)
        for rank, (hyp, count) in enumerate(counts.head(top_k).items(), 1):
            h = work[work["hyp_norm_v2"] == hyp]
            rows.append(
                {
                    "condition": condition,
                    "rank": rank,
                    "normalized_hypothesis": hyp,
                    "example_raw_hypothesis": str(h.iloc[0]["hypothesis"]),
                    "count": int(count),
                    "mass": float(count / len(work)),
                    "WER_mean": float(h["WER"].mean(skipna=True)),
                    "qwen_plaus_mean": float(h["qwen_plaus"].astype(float).mean()),
                    "gpt2_plaus_mean": float(h["gpt2_plaus"].astype(float).mean()),
                    "strict_H_qwen_frac": float(h["strict_h_qwen"].astype(bool).mean()),
                    "strict_H_gpt2_frac": float(h["strict_h_gpt2"].astype(bool).mean()),
                    "ctc_support_nll_mean": float(h["ctc_support_nll"].replace([np.inf, -np.inf], np.nan).mean()),
                    "gate_accept_frac": float(h["gate_accepted"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "top_hypotheses_by_condition.csv", index=False)
    return result


def save_clean_candidates(test: pd.DataFrame, tau: float, outdir: Path) -> pd.DataFrame:
    clean = test[test["perturbation"] == "none"].copy()
    clean["gate_accepted"] = accepted_mask(clean["ctc_support_nll"].astype(float), tau)
    clean["gate_caught"] = ~clean["gate_accepted"]
    clean["strict_h_union"] = clean["strict_h_qwen"] | clean["strict_h_gpt2"]
    clean["strict_h_both"] = clean["strict_h_qwen"] & clean["strict_h_gpt2"]
    candidates = clean[clean["strict_h_union"]].copy()
    candidates["lm_membership"] = np.where(
        candidates["strict_h_both"],
        "both",
        np.where(candidates["strict_h_qwen"], "qwen", "gpt2"),
    )
    candidates = candidates.sort_values(["strict_h_both", "WER"], ascending=[False, False])
    candidates.to_csv(outdir / "strict_clean_hallucination_candidates_corrected.csv", index=False)
    return candidates


def main() -> None:
    p = argparse.ArgumentParser(description="Correct WER and explore pretrained Whisper stress outputs")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--strict_wer", type=float, default=0.5)
    p.add_argument("--min_clean_coverage", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.source)
    required = {
        "split", "perturbation", "reference", "hypothesis", "qwen_plaus",
        "gpt2_plaus", "ctc_support_nll",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rescored = add_clean_wer_v2(df)
    thresholds = derive_thresholds(rescored, args.strict_wer)
    rescored = apply_labels(rescored, thresholds)

    dev = rescored[rescored["split"] == "dev"].copy()
    test = rescored[rescored["split"] == "test"].copy()
    gate_dev = dev.copy()
    gate_dev["hallucination_like_qwen"] = gate_dev["strict_h_qwen"].astype(bool)
    gate_dev["hallucination_like_gpt2"] = gate_dev["strict_h_gpt2"].astype(bool)
    tau, calibration = select_gate_threshold(gate_dev, min_clean_coverage=args.min_clean_coverage)

    rescored.to_csv(outdir / "scored_outputs_corrected.csv", index=False)
    calibration.to_csv(outdir / "gate_calibration_corrected.csv", index=False)
    (outdir / "frozen_hallucination_thresholds_corrected.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n"
    )
    (outdir / "frozen_gate_threshold_corrected.json").write_text(
        json.dumps(
            {
                "threshold": float(tau),
                "selection_split": "dev",
                "target_label": "strict_h_qwen",
                "strict_wer_threshold": float(args.strict_wer),
                "min_clean_dev_coverage": float(args.min_clean_coverage),
                "reference_free_at_test": True,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    dev_summary = summarize_conditions(rescored, "dev")
    test_summary = summarize_conditions(rescored, "test")
    gate_summary = gate_test_summary(test, tau)
    dev_summary.to_csv(outdir / "dev_condition_summary_corrected.csv", index=False)
    test_summary.to_csv(outdir / "test_condition_summary_corrected.csv", index=False)
    gate_summary.to_csv(outdir / "test_gate_summary_corrected.csv", index=False)

    candidates = save_clean_candidates(test, tau, outdir)
    tops = save_top_hypotheses(test, tau, outdir, args.top_k)

    changed = rescored[rescored["WER_changed"]].copy()
    changed.to_csv(outdir / "rows_changed_by_wer_fix.csv", index=False)
    clean_test = test[test["perturbation"] == "none"].copy()
    old_strict_union = (
        (pd.to_numeric(clean_test["WER_previous"], errors="coerce") > args.strict_wer)
        & (
            (clean_test["qwen_plaus"].astype(float) > thresholds["qwen_plausibility_threshold"])
            | (clean_test["gpt2_plaus"].astype(float) > thresholds["gpt2_plausibility_threshold"])
        )
    )
    new_strict_union = clean_test["strict_h_qwen"] | clean_test["strict_h_gpt2"]

    report = {
        "source": str(args.source),
        "normalization_bug_fixed": "old optional-apostrophe n't regex corrupted words ending in nt",
        "rows_total": int(len(rescored)),
        "rows_with_changed_WER": int(rescored["WER_changed"].sum()),
        "clean_test_rows_with_changed_WER": int(clean_test["WER_changed"].sum()),
        "clean_test_WER_mean_corrected": float(clean_test["WER"].mean(skipna=True)),
        "clean_test_strict_union_previous_using_new_lm_thresholds": int(old_strict_union.sum()),
        "clean_test_strict_union_corrected": int(new_strict_union.sum()),
        "gate_tau_corrected": float(tau),
        "strict_clean_candidates_corrected": int(len(candidates)),
    }
    (outdir / "rescore_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("=== Corrected pretrained Whisper WER + failure exploration ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nTEST summary:")
    print(test_summary.to_string(index=False))
    print("\nGate summary:")
    print(gate_summary.to_string(index=False))
    print("\nTop hypotheses (first 10 rows):")
    print(tops.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
