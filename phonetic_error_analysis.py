#!/usr/bin/env python3
"""Text-only phonetic audit of cached ASR candidates.

Broad candidate = WER > 0.1314136929820308 and Qwen plausibility > 0.8658364617260637.
Strict candidate = WER > 0.50 with the same Qwen threshold.

Reference and hypothesis are converted to stress-free CMUdict ARPABET and scored
with normalized phone edit distance (NPED):
    edit_distance(ref_phones, hyp_phones) / max(len(ref_phones), len(hyp_phones)).
The primary "phonetically explainable" cutoff is NPED <= 0.30; 0.20/0.25/0.30/0.35
are always reported as a sensitivity analysis. Residuals are called non-phonetic
candidates, not human-verified hallucinations.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cmudict
import numpy as np
import pandas as pd

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_INPUT = ROOT / "training_error_risk_map" / "per_utterance_training_error_risk_map.csv"
DEFAULT_OUTPUT = ROOT / "training_error_risk_map" / "phonetic_analysis"
WER_THR = 0.1314136929820308
QWEN_THR = 0.8658364617260637
STRICT_WER_THR = 0.50
SENSITIVITY = (0.20, 0.25, 0.30, 0.35)
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STRESS_RE = re.compile(r"\d")


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def encode(text, lex):
    words = WORD_RE.findall(str(text).lower())
    phones, covered = [], 0
    for word in words:
        prons = lex.get(word)
        if not prons:
            continue
        phones.extend(STRESS_RE.sub("", p) for p in prons[0])
        covered += 1
    coverage = covered / len(words) if words else 0.0
    return phones, len(words), coverage


def phone_row(reference, hypothesis, lex):
    ref, ref_words, ref_cov = encode(reference, lex)
    hyp, hyp_words, hyp_cov = encode(hypothesis, lex)
    if not ref and not hyp:
        dist, nped = 0, np.nan
    else:
        dist = edit_distance(ref, hyp)
        nped = dist / max(len(ref), len(hyp), 1)
    return {
        "reference_word_count": ref_words,
        "hypothesis_word_count": hyp_words,
        "reference_phone_coverage": ref_cov,
        "hypothesis_phone_coverage": hyp_cov,
        "reference_phone_count": len(ref),
        "hypothesis_phone_count": len(hyp),
        "phone_edit_distance": dist,
        "normalized_phone_edit_distance": nped,
        "phone_similarity": 1.0 - nped if np.isfinite(nped) else np.nan,
    }


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(df, primary):
    group_cols = [c for c in ["model_name", "condition", "corruption_ratio", "perturbation"] if c in df]
    groups = df.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), df)]
    rows, sens = [], []
    ps = f"{int(round(primary * 100)):02d}"
    for key, g in groups:
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            meta = dict(zip(group_cols, key))
        else:
            meta = {}
        valid = g.valid_phone_comparison.astype(bool)
        broad = g.candidate_broad.astype(bool)
        strict = g.candidate_strict.astype(bool)
        bv, sv = broad & valid, strict & valid
        bn = g[f"broad_nonphonetic_{ps}"].astype(bool)
        sn = g[f"strict_nonphonetic_{ps}"].astype(bool)
        bvn, svn = int(bv.sum()), int(sv.sum())
        blo, bhi = wilson(int(bn.sum()), bvn)
        slo, shi = wilson(int(sn.sum()), svn)
        rows.append({
            **meta,
            "N": len(g),
            "valid_phone_rate_pct": 100 * valid.mean(),
            "broad_candidate_rate_pct": 100 * broad.mean(),
            "broad_candidate_valid_N": bvn,
            "broad_candidate_median_NPED": g.loc[bv, "normalized_phone_edit_distance"].median(),
            "broad_phonetic_explainable_pct": 100 * (1 - bn.sum() / bvn) if bvn else np.nan,
            "broad_nonphonetic_pct_of_valid_candidates": 100 * bn.sum() / bvn if bvn else np.nan,
            "broad_nonphonetic_rate_pct_all": 100 * bn.mean(),
            "broad_nonphonetic_95ci_low_pct": 100 * blo,
            "broad_nonphonetic_95ci_high_pct": 100 * bhi,
            "strict_candidate_rate_pct": 100 * strict.mean(),
            "strict_candidate_valid_N": svn,
            "strict_candidate_median_NPED": g.loc[sv, "normalized_phone_edit_distance"].median(),
            "strict_nonphonetic_rate_pct_all": 100 * sn.mean(),
            "strict_nonphonetic_95ci_low_pct": 100 * slo,
            "strict_nonphonetic_95ci_high_pct": 100 * shi,
        })
        for tau in SENSITIVITY:
            s = f"{int(round(tau * 100)):02d}"
            b = g[f"broad_nonphonetic_{s}"].astype(bool)
            st = g[f"strict_nonphonetic_{s}"].astype(bool)
            sens.append({
                **meta,
                "phone_threshold": tau,
                "broad_valid_candidate_N": bvn,
                "broad_nonphonetic_pct_of_valid_candidates": 100 * b.sum() / bvn if bvn else np.nan,
                "broad_nonphonetic_rate_pct_all": 100 * b.mean(),
                "strict_valid_candidate_N": svn,
                "strict_nonphonetic_rate_pct_all": 100 * st.mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(sens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min_word_coverage", type=float, default=0.80)
    p.add_argument("--primary_phone_threshold", type=float, default=0.30)
    a = p.parse_args()
    if not a.input.exists():
        raise FileNotFoundError(a.input)
    if not 0 <= a.min_word_coverage <= 1 or not 0 <= a.primary_phone_threshold <= 1:
        raise ValueError("coverage and phone thresholds must be in [0,1]")

    df = pd.read_csv(a.input)
    for col in ["reference", "hypothesis"]:
        if col not in df:
            raise ValueError(f"Missing required column: {col}")
    wer_col = "WER" if "WER" in df else "wer"
    qwen_col = "qwen_plaus" if "qwen_plaus" in df else "normalized_sentence_score_Qwen3-0.6B"
    if wer_col not in df or qwen_col not in df:
        raise ValueError("Missing WER or Qwen plausibility column")

    wer = pd.to_numeric(df[wer_col], errors="coerce")
    qwen = pd.to_numeric(df[qwen_col], errors="coerce")
    df["candidate_broad"] = (wer > WER_THR) & (qwen > QWEN_THR)
    df["candidate_strict"] = (wer > STRICT_WER_THR) & (qwen > QWEN_THR)

    lex = {str(k).lower(): v for k, v in cmudict.dict().items()}
    ph = pd.DataFrame([phone_row(r, h, lex) for r, h in zip(df.reference, df.hypothesis)], index=df.index)
    df = pd.concat([df, ph], axis=1)
    df["valid_phone_comparison"] = (
        (df.reference_word_count > 0) & (df.hypothesis_word_count > 0)
        & (df.reference_phone_coverage >= a.min_word_coverage)
        & (df.hypothesis_phone_coverage >= a.min_word_coverage)
        & np.isfinite(df.normalized_phone_edit_distance)
    )
    for tau in sorted(set(SENSITIVITY + (a.primary_phone_threshold,))):
        s = f"{int(round(tau * 100)):02d}"
        explainable = df.valid_phone_comparison & (df.normalized_phone_edit_distance <= tau)
        df[f"phonetic_explainable_{s}"] = explainable
        df[f"broad_nonphonetic_{s}"] = df.candidate_broad & df.valid_phone_comparison & ~explainable
        df[f"strict_nonphonetic_{s}"] = df.candidate_strict & df.valid_phone_comparison & ~explainable

    summary, sensitivity = summarize(df, a.primary_phone_threshold)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output_dir / "per_utterance_phonetic_analysis.csv", index=False)
    summary.to_csv(a.output_dir / "summary_phonetic_analysis.csv", index=False)
    sensitivity.to_csv(a.output_dir / "sensitivity_phonetic_thresholds.csv", index=False)
    manifest = {
        "input": str(a.input), "wer_threshold": WER_THR,
        "qwen_plausibility_threshold": QWEN_THR, "strict_wer_threshold": STRICT_WER_THR,
        "phone_representation": "CMUdict ARPABET, lexical stress removed, first pronunciation",
        "phone_metric": "Levenshtein/max phone-sequence length",
        "min_word_coverage": a.min_word_coverage,
        "primary_phone_threshold": a.primary_phone_threshold,
        "sensitivity_thresholds": SENSITIVITY,
        "interpretation_guard": "Residuals are non-phonetic candidates, not human-verified hallucinations."
    }
    (a.output_dir / "phonetic_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cols = [c for c in ["model_name", "condition", "corruption_ratio", "perturbation", "N",
            "valid_phone_rate_pct", "broad_candidate_rate_pct", "broad_candidate_median_NPED",
            "broad_phonetic_explainable_pct", "broad_nonphonetic_pct_of_valid_candidates",
            "broad_nonphonetic_rate_pct_all", "strict_candidate_rate_pct",
            "strict_nonphonetic_rate_pct_all"] if c in summary]
    print("=== Phonetic-error audit ===")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Threshold sensitivity ===")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
