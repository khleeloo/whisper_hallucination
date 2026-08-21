#!/usr/bin/env python3
"""Recompute paper-facing WER and hallucination scoring with cleaned ASR text normalization.

This script deliberately reuses cached Whisper hypotheses, Qwen3/GPT-2 plausibility
scores, and wav2vec2-CTC support scores.  Only the WER/reference-dependent part of
the evaluation is corrected, so changes in hallucination labels can be attributed
to WER normalization rather than to rerunning the models.

The previous normalization deleted punctuation.  That can create artificial word
errors for contractions and hyphenated compounds (e.g. ``doesn't`` vs ``does not``
or ``African-American`` vs ``african american``).  The cleaned normalization:

  * Unicode-normalizes and lowercases;
  * expands unambiguous English contractions (n't, 'll, 're, 've, 'm);
  * handles already-split contraction fragments such as ``she ll``;
  * turns hyphens/dashes and all remaining punctuation into token boundaries;
  * collapses whitespace;
  * treats empty/common placeholder references as invalid for WER/H labels.

Outputs include corrected scored rows, frozen clean-DEV thresholds, held-out TEST
mitigation summaries, a clean risk/coverage sweep, before/caught/missed clean H
files at the DEV-95%-coverage operating point, and (when available) a rescored
canonical acoustic-stress table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_SOURCE = ROOT / "hallucination_mitigation_acoustic_before_after" / "scored_outputs.csv"
DEFAULT_STRESS_SOURCE = ROOT / "acoustic_stress_full" / "per_utterance_acoustic_stress.csv"
DEFAULT_OUTPUT = ROOT / "clean_wer_rescore"
DEFAULT_COVERAGES = [0.99, 0.98, 0.95, 0.90, 0.85]
CLEAN = "none"
FULL_05 = "full_noise_amp0.5_dur0.0"
FULL_075 = "full_noise_amp0.75_dur0.0"
INVALID_REFERENCES = {"", "undefined", "none", "null", "nan", "n/a", "na"}
WHISPER_SPECIAL = re.compile(r"<\|[^|]+\|>")


def normalize_asr_text(text: object) -> str:
    """Normalize reference/hypothesis text for token-level ASR WER.

    This is intentionally conservative: formatting/token-boundary differences are
    removed, but lexical alternatives (e.g. ``logout`` vs ``log out``) remain
    errors.  Ambiguous 's/'d contractions are not expanded.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = WHISPER_SPECIAL.sub(" ", s)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")

    # Unambiguous/common contractions. Do these before punctuation removal.
    s = re.sub(r"\bwon't\b", "will not", s)
    s = re.sub(r"\bcan't\b", "can not", s)
    s = re.sub(r"\bshan't\b", "shall not", s)
    s = re.sub(r"n['’]?t\b", " not", s)
    s = re.sub(r"'ll\b", " will", s)
    s = re.sub(r"'re\b", " are", s)
    s = re.sub(r"'ve\b", " have", s)
    s = re.sub(r"'m\b", " am", s)

    # Hyphenation is formatting for WER: cigar-making == cigar making.
    s = re.sub(r"[-‐‑‒–—―/]+", " ", s)
    # Remaining punctuation/symbols become boundaries rather than being deleted.
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Some decoder outputs already contain split contraction fragments without
    # apostrophes. Normalize only unambiguous pronoun forms.
    s = re.sub(r"\b(i|you|he|she|it|we|they)\s+ll\b", r"\1 will", s)
    s = re.sub(r"\b(you|we|they)\s+re\b", r"\1 are", s)
    s = re.sub(r"\b(i|you|we|they)\s+ve\b", r"\1 have", s)
    s = re.sub(r"\bi\s+m\b", "i am", s)
    s = re.sub(r"\b(can)\s+n\s+t\b", r"\1 not", s)
    s = re.sub(r"\b(will)\s+n\s+t\b", r"\1 not", s)
    s = re.sub(r"\b(\w+)\s+n\s+t\b", r"\1 not", s)
    return re.sub(r"\s+", " ", s).strip()


def _levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
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


def clean_wer(reference: object, hypothesis: object) -> Tuple[float, str, str, bool]:
    ref = normalize_asr_text(reference)
    hyp = normalize_asr_text(hypothesis)
    valid = ref not in INVALID_REFERENCES
    if not valid:
        return float("nan"), ref, hyp, False
    ref_words = ref.split()
    hyp_words = hyp.split()
    return float(_levenshtein_distance(ref_words, hyp_words) / len(ref_words)), ref, hyp, True


def add_clean_wer(df: pd.DataFrame) -> pd.DataFrame:
    required = {"reference", "hypothesis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for WER rescoring: {sorted(missing)}")
    out = df.copy()
    if "WER" in out.columns:
        out["WER_legacy"] = pd.to_numeric(out["WER"], errors="coerce")
    rows = [clean_wer(r, h) for r, h in zip(out["reference"], out["hypothesis"])]
    out["WER"] = [x[0] for x in rows]
    out["reference_norm_cleanwer"] = [x[1] for x in rows]
    out["hypothesis_norm_cleanwer"] = [x[2] for x in rows]
    out["valid_reference_cleanwer"] = [x[3] for x in rows]
    out["reference_words_cleanwer"] = out["reference_norm_cleanwer"].map(lambda x: len(str(x).split()))
    out["hypothesis_words_cleanwer"] = out["hypothesis_norm_cleanwer"].map(lambda x: len(str(x).split()))
    return out


def derive_dev_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    clean = df[(df["split"].astype(str) == "dev") & (df["perturbation"].astype(str) == CLEAN)]
    clean = clean[clean["valid_reference_cleanwer"].astype(bool) & np.isfinite(clean["WER"].astype(float))]
    if clean.empty:
        raise ValueError("No valid clean DEV rows for threshold derivation")
    return {
        "wer_threshold": float(clean["WER"].mean()),
        "qwen_plausibility_threshold": float(clean["qwen_plaus"].astype(float).mean()),
        "gpt2_plausibility_threshold": float(clean["gpt2_plaus"].astype(float).mean()),
        "N_clean_dev_valid": int(len(clean)),
    }


def apply_labels(df: pd.DataFrame, thresholds: Dict[str, float], *, prefix: str = "hallucination_like") -> pd.DataFrame:
    out = df.copy()
    valid = out["valid_reference_cleanwer"].astype(bool) & np.isfinite(out["WER"].astype(float))
    high_wer = valid & (out["WER"].astype(float) > thresholds["wer_threshold"])
    out[f"{prefix}_qwen"] = high_wer & (out["qwen_plaus"].astype(float) > thresholds["qwen_plausibility_threshold"])
    out[f"{prefix}_gpt2"] = high_wer & (out["gpt2_plaus"].astype(float) > thresholds["gpt2_plausibility_threshold"])
    out[f"{prefix}_both"] = out[f"{prefix}_qwen"] & out[f"{prefix}_gpt2"]
    out[f"{prefix}_union"] = out[f"{prefix}_qwen"] | out[f"{prefix}_gpt2"]
    return out


def accepted(scores: Iterable[float], tau: float) -> np.ndarray:
    x = np.asarray(list(scores), dtype=float)
    return np.isfinite(x) & (x <= float(tau))


def threshold_for_coverage(scores: Sequence[float], target: float) -> Tuple[float, float]:
    x = np.asarray(scores, dtype=float)
    finite = np.sort(x[np.isfinite(x)])
    need = int(math.ceil(target * len(x) - 1e-12))
    if need < 1 or len(finite) < need:
        raise ValueError(f"Cannot achieve target coverage {target:.3f}")
    tau = float(finite[need - 1])
    return tau, float(accepted(x, tau).mean())


def summarize_gate(group: pd.DataFrame, tau: float) -> Dict[str, object]:
    acc = accepted(group["ctc_support_nll"], tau)
    row: Dict[str, object] = {
        "condition": str(group.iloc[0]["perturbation"]),
        "N": int(len(group)),
        "coverage": float(acc.mean()),
        "abstention": float((~acc).mean()),
        "WER_before": float(group["WER"].mean(skipna=True)),
        "WER_emitted": float(group.loc[acc, "WER"].mean(skipna=True)) if acc.any() else float("nan"),
    }
    for lm in ("qwen", "gpt2"):
        h = group[f"hallucination_like_{lm}"].astype(bool).to_numpy()
        emitted = h & acc
        n_h = int(h.sum())
        row[f"{lm}_H_before"] = float(h.mean())
        row[f"{lm}_H_after_system"] = float(emitted.mean())
        row[f"{lm}_H_among_emitted"] = float(emitted.sum() / acc.sum()) if acc.sum() else float("nan")
        row[f"{lm}_H_capture"] = float((h & ~acc).sum() / n_h) if n_h else 0.0
        row[f"{lm}_rejection_precision"] = float((h & ~acc).sum() / (~acc).sum()) if (~acc).sum() else float("nan")
    return row


def risk_coverage(df: pd.DataFrame, coverages: Sequence[float]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    clean_dev = df[(df["split"] == "dev") & (df["perturbation"] == CLEAN)]
    test = df[df["split"] == "test"]
    rows: List[Dict[str, object]] = []
    taus: Dict[str, float] = {}
    for cov in coverages:
        tau, realized = threshold_for_coverage(clean_dev["ctc_support_nll"].astype(float), cov)
        name = f"clean_cov_{cov:.3f}"
        taus[name] = tau
        for condition in (CLEAN, FULL_05, FULL_075):
            g = test[test["perturbation"] == condition]
            if g.empty:
                continue
            r = summarize_gate(g, tau)
            r.update({"operating_point": name, "target_clean_dev_coverage": cov, "realized_clean_dev_coverage": realized, "threshold": tau})
            rows.append(r)
    return pd.DataFrame(rows), taus


def save_best_gate_hallucinations(df: pd.DataFrame, tau: float, outdir: Path) -> None:
    clean = df[(df["split"] == "test") & (df["perturbation"] == CLEAN)].copy()
    clean["accepted_best_gate"] = accepted(clean["ctc_support_nll"], tau)
    clean["abstained_best_gate"] = ~clean["accepted_best_gate"]
    union = clean[clean["hallucination_like_union"]].copy()
    union.to_csv(outdir / "clean_test_hallucinations_union_cleanwer.csv", index=False)
    for lm in ("qwen", "gpt2"):
        h = clean[clean[f"hallucination_like_{lm}"]].copy()
        h.to_csv(outdir / f"{lm}_hallucinations_before_cleanwer.csv", index=False)
        h[h["abstained_best_gate"]].to_csv(outdir / f"{lm}_hallucinations_caught_cleanwer.csv", index=False)
        h[h["accepted_best_gate"]].to_csv(outdir / f"{lm}_hallucinations_missed_cleanwer.csv", index=False)


def rescore_stress(path: Path, outdir: Path) -> None:
    if not path.exists():
        print(f"Stress source not found; skipping: {path}", flush=True)
        return
    stress = pd.read_csv(path)
    needed = {"reference", "hypothesis", "perturbation", "qwen_plaus", "gpt2_plaus"}
    missing = needed - set(stress.columns)
    if missing:
        print(f"Stress source missing columns {sorted(missing)}; skipping", flush=True)
        return
    stress = add_clean_wer(stress)
    clean = stress[(stress["perturbation"] == CLEAN) & stress["valid_reference_cleanwer"]]
    th = {
        "wer_threshold": float(clean["WER"].mean()),
        "qwen_plausibility_threshold": float(clean["qwen_plaus"].astype(float).mean()),
        "gpt2_plausibility_threshold": float(clean["gpt2_plaus"].astype(float).mean()),
        "N_clean_valid": int(len(clean)),
    }
    stress = apply_labels(stress, th, prefix="hallucination")
    stress.to_csv(outdir / "acoustic_stress_per_utterance_cleanwer.csv", index=False)
    rows = []
    for condition, g in stress.groupby("perturbation", sort=False):
        valid = g["valid_reference_cleanwer"].astype(bool)
        gv = g[valid]
        rows.append({
            "condition": condition,
            "N": int(len(g)),
            "N_valid_reference": int(valid.sum()),
            "WER": float(gv["WER"].mean()) if len(gv) else float("nan"),
            "qwen_plaus": float(g["qwen_plaus"].astype(float).mean()),
            "gpt2_plaus": float(g["gpt2_plaus"].astype(float).mean()),
            "qwen_H_pct": 100.0 * float(g["hallucination_qwen"].astype(bool).mean()),
            "gpt2_H_pct": 100.0 * float(g["hallucination_gpt2"].astype(bool).mean()),
        })
    pd.DataFrame(rows).to_csv(outdir / "acoustic_stress_summary_cleanwer.csv", index=False)
    (outdir / "acoustic_stress_thresholds_cleanwer.json").write_text(json.dumps(th, indent=2, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Clean WER + cached dual-LM hallucination rescoring")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--stress_source", type=Path, default=DEFAULT_STRESS_SOURCE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.source)
    required = {"split", "perturbation", "reference", "hypothesis", "qwen_plaus", "gpt2_plaus", "ctc_support_nll"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Source is missing required columns: {sorted(missing)}")

    rescored = add_clean_wer(raw)
    thresholds = derive_dev_thresholds(rescored)
    rescored = apply_labels(rescored, thresholds)
    rescored.to_csv(args.output_dir / "scored_outputs_cleanwer.csv", index=False)
    (args.output_dir / "frozen_hallucination_thresholds_cleanwer.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # 98%-DEV gate summary, matching the original paper-facing mitigation setup.
    clean_dev = rescored[(rescored["split"] == "dev") & (rescored["perturbation"] == CLEAN)]
    tau98, realized98 = threshold_for_coverage(clean_dev["ctc_support_nll"].astype(float), 0.98)
    test = rescored[rescored["split"] == "test"]
    summary_rows = [summarize_gate(g, tau98) for _, g in test.groupby("perturbation", sort=False)]
    summary = pd.DataFrame(summary_rows)
    summary["threshold"] = tau98
    summary["target_clean_dev_coverage"] = 0.98
    summary["realized_clean_dev_coverage"] = realized98
    summary.to_csv(args.output_dir / "test_mitigation_summary_cleanwer.csv", index=False)

    sweep, taus = risk_coverage(rescored, args.coverages)
    sweep.to_csv(args.output_dir / "clean_transfer_risk_coverage_cleanwer.csv", index=False)

    best_name = "clean_cov_0.950"
    if best_name in taus:
        save_best_gate_hallucinations(rescored, taus[best_name], args.output_dir)

    # Quantify how much the normalization changed clean WER directly.
    clean_test = rescored[(rescored["split"] == "test") & (rescored["perturbation"] == CLEAN)]
    norm_report = {
        "N_clean_test": int(len(clean_test)),
        "N_valid_reference_clean_test": int(clean_test["valid_reference_cleanwer"].sum()),
        "legacy_mean_WER_clean_test": float(clean_test["WER_legacy"].mean()) if "WER_legacy" in clean_test else None,
        "clean_mean_WER_clean_test": float(clean_test["WER"].mean(skipna=True)),
        "legacy_median_WER_clean_test": float(clean_test["WER_legacy"].median()) if "WER_legacy" in clean_test else None,
        "clean_median_WER_clean_test": float(clean_test["WER"].median(skipna=True)),
        "exact_after_clean_normalization_pct": 100.0 * float((clean_test["WER"].fillna(np.inf) == 0).mean()),
        "invalid_reference_ids": clean_test.loc[~clean_test["valid_reference_cleanwer"], "utterance_id"].astype(str).tolist(),
        "dev_thresholds": thresholds,
        "tau_98pct_clean_dev": tau98,
    }
    (args.output_dir / "normalization_report.json").write_text(json.dumps(norm_report, indent=2, sort_keys=True) + "\n")

    rescore_stress(args.stress_source, args.output_dir)

    print("=== CLEAN WER RESCORE COMPLETE ===")
    print(json.dumps(norm_report, indent=2))
    print("\nHeld-out TEST mitigation with corrected WER:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClean TEST risk/coverage:")
    print(sweep[sweep["condition"] == CLEAN].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
