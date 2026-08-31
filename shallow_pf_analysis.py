#!/usr/bin/env python3
"""Run the official SHALLOW phonetic-fabrication (PF) formulation on cached ASR outputs.

This is the exact PF formulation used by SALT-Research/SHALLOW at pinned commit
47de7b6646a1a18735bf0860efa84f70d2e2ef06 (src/fabrications.py and
src/shallow.py):
  1) Metaphone-encode reference and hypothesis with jellyfish.metaphone.
  2) Compute Hamming distance / max encoded length.
  3) Compute Levenshtein distance / max encoded length.
  4) Compute Jaro-Winkler similarity.
  5) PF = (hamming_norm + levenshtein_norm + (1 - jaro_winkler)) / 3.

Higher PF means greater phonetic fabrication / lower phonetic similarity.
SHALLOW does not define a binary PF hallucination threshold, so this script does
not invent one. It reports PF distributions for all outputs and for the frozen
high-error/high-Qwen-plausibility candidate subsets used in our paper.

Pairwise confidence intervals use a matched-utterance bootstrap. The same held-out
utterance IDs are resampled jointly across the two compared corruption conditions.
For candidate-only subsets, the condition-specific candidate masks are reapplied
inside every paired bootstrap resample, so the marginal subset means are preserved
without breaking the matched evaluation design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd

SHALLOW_REPO = "https://github.com/SALT-Research/SHALLOW"
SHALLOW_COMMIT = "47de7b6646a1a18735bf0860efa84f70d2e2ef06"
SHALLOW_FABRICATIONS_PATH = "src/fabrications.py"
SHALLOW_AGGREGATION_PATH = "src/shallow.py"
JELLYFISH_VERSION = "1.2.0"

WER_THRESHOLD = 0.1314136929820308
QWEN_THRESHOLD = 0.8658364617260637
STRICT_WER_THRESHOLD = 0.50

CONDITION_NAMES = {
    "RR": "Repeated pairs",
    "RU": "Repeated audio",
    "UR": "Repeated target",
    "UU": "Random mismatch",
}


def shallow_phonetic_components(reference: str, hypothesis: str) -> tuple[float, float, float, float, str, str]:
    """Exact SHALLOW PF component computation."""
    reference = "" if pd.isna(reference) else str(reference)
    hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)
    if reference == hypothesis:
        return 0.0, 0.0, 1.0, 0.0, jellyfish.metaphone(reference), jellyfish.metaphone(hypothesis)
    ref_meta = jellyfish.metaphone(reference)
    hyp_meta = jellyfish.metaphone(hypothesis)
    hamm = jellyfish.hamming_distance(ref_meta, hyp_meta)
    max_h = max(len(ref_meta), len(hyp_meta), 1)
    hamm_norm = hamm / max_h if hamm is not None else 0.0
    leven = jellyfish.levenshtein_distance(ref_meta, hyp_meta)
    max_l = max(len(ref_meta), len(hyp_meta), 1)
    leven_norm = leven / max_l if leven is not None else 0.0
    jaro_winkler = jellyfish.jaro_winkler_similarity(ref_meta, hyp_meta)
    pf = (hamm_norm + leven_norm + (1.0 - jaro_winkler)) / 3.0
    return float(hamm_norm), float(leven_norm), float(jaro_winkler), float(pf), ref_meta, hyp_meta


def describe_pf(values: pd.Series, prefix: str) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {f"{prefix}_N": 0, f"{prefix}_PF_mean": np.nan, f"{prefix}_PF_median": np.nan, f"{prefix}_PF_q25": np.nan, f"{prefix}_PF_q75": np.nan}
    return {
        f"{prefix}_N": int(len(x)),
        f"{prefix}_PF_mean": float(np.mean(x)),
        f"{prefix}_PF_median": float(np.median(x)),
        f"{prefix}_PF_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_PF_q75": float(np.quantile(x, 0.75)),
    }


def _subset_mean(frame: pd.DataFrame, pf_col: str, mask_col: str | None) -> float:
    if mask_col is None:
        values = pd.to_numeric(frame[pf_col], errors="coerce")
    else:
        mask = pd.to_numeric(frame[mask_col], errors="coerce").fillna(0).astype(int) == 1
        values = pd.to_numeric(frame.loc[mask, pf_col], errors="coerce")
    values = values.dropna()
    return float(values.mean()) if len(values) else np.nan


def paired_bootstrap_mean_contrast(
    paired: pd.DataFrame,
    mask_base: str | None,
    seed: int,
    reps: int = 10000,
) -> tuple[float, float, float, int, int, int]:
    """Matched-utterance bootstrap CI for mean PF difference A-B."""
    if len(paired) == 0:
        return np.nan, np.nan, np.nan, 0, 0, 0
    mask_a = None if mask_base is None else f"{mask_base}_a"
    mask_b = None if mask_base is None else f"{mask_base}_b"
    mean_a = _subset_mean(paired, "shallow_PF_a", mask_a)
    mean_b = _subset_mean(paired, "shallow_PF_b", mask_b)
    obs = mean_a - mean_b if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan
    if mask_a is None:
        n_a = int(pd.to_numeric(paired["shallow_PF_a"], errors="coerce").notna().sum())
        n_b = int(pd.to_numeric(paired["shallow_PF_b"], errors="coerce").notna().sum())
    else:
        valid_a = pd.to_numeric(paired["shallow_PF_a"], errors="coerce").notna()
        valid_b = pd.to_numeric(paired["shallow_PF_b"], errors="coerce").notna()
        n_a = int(((pd.to_numeric(paired[mask_a], errors="coerce").fillna(0) == 1) & valid_a).sum())
        n_b = int(((pd.to_numeric(paired[mask_b], errors="coerce").fillna(0) == 1) & valid_b).sum())
    rng = np.random.default_rng(seed)
    diffs = np.full(reps, np.nan, dtype=float)
    n = len(paired)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        boot = paired.iloc[idx]
        boot_a = _subset_mean(boot, "shallow_PF_a", mask_a)
        boot_b = _subset_mean(boot, "shallow_PF_b", mask_b)
        if np.isfinite(boot_a) and np.isfinite(boot_b):
            diffs[i] = boot_a - boot_b
    valid_diffs = diffs[np.isfinite(diffs)]
    if len(valid_diffs) == 0:
        return obs, np.nan, np.nan, n_a, n_b, 0
    lo, hi = np.quantile(valid_diffs, [0.025, 0.975])
    return obs, float(lo), float(hi), n_a, n_b, int(len(valid_diffs))


def make_paired_conditions(pg: pd.DataFrame, ca: str, cb: str) -> pd.DataFrame:
    cols = ["utterance_id", "shallow_PF", "broad_candidate", "broad_nonrep_candidate"]
    a = pg.loc[pg["condition"] == ca, cols].copy()
    b = pg.loc[pg["condition"] == cb, cols].copy()
    if a["utterance_id"].duplicated().any() or b["utterance_id"].duplicated().any():
        raise ValueError(f"Duplicate utterance IDs prevent one-to-one pairing for {ca} vs {cb}.")
    paired = a.merge(b, on="utterance_id", how="inner", suffixes=("_a", "_b"), validate="one_to_one")
    if len(paired) != len(a) or len(paired) != len(b):
        raise ValueError(f"Unmatched utterance IDs for {ca} vs {cb}: A={len(a)}, B={len(b)}, matched={len(paired)}")
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    required = {"utterance_id", "reference", "hypothesis", "WER", "qwen_plaus", "model_name", "condition", "corruption_ratio", "perturbation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    components = [shallow_phonetic_components(r, h) for r, h in zip(df["reference"], df["hypothesis"])]
    df["shallow_phonetic_hamming"] = [x[0] for x in components]
    df["shallow_phonetic_levenshtein"] = [x[1] for x in components]
    df["shallow_phonetic_jaro_winkler"] = [x[2] for x in components]
    df["shallow_PF"] = [x[3] for x in components]
    df["shallow_ref_metaphone"] = [x[4] for x in components]
    df["shallow_hyp_metaphone"] = [x[5] for x in components]
    df["condition_name"] = df["condition"].map(CONDITION_NAMES).fillna(df["condition"].astype(str))

    wer = pd.to_numeric(df["WER"], errors="coerce")
    qwen = pd.to_numeric(df["qwen_plaus"], errors="coerce")
    df["broad_candidate"] = ((wer > WER_THRESHOLD) & (qwen > QWEN_THRESHOLD)).astype(int)
    df["strict_candidate"] = ((wer > STRICT_WER_THRESHOLD) & (qwen > QWEN_THRESHOLD)).astype(int)
    if "rep34" in df.columns:
        df["broad_nonrep_candidate"] = ((df["broad_candidate"] == 1) & (pd.to_numeric(df["rep34"], errors="coerce").fillna(0) == 0)).astype(int)
    else:
        df["broad_nonrep_candidate"] = df["broad_candidate"]

    per_path = args.output_dir / "per_utterance_shallow_pf.csv"
    df.to_csv(per_path, index=False)

    rows = []
    group_cols = ["model_name", "condition", "condition_name", "corruption_ratio", "perturbation"]
    for keys, g in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["N"] = int(len(g))
        row.update(describe_pf(g["shallow_PF"], "all"))
        row.update(describe_pf(g.loc[g["broad_candidate"] == 1, "shallow_PF"], "broad_candidate"))
        row.update(describe_pf(g.loc[g["broad_nonrep_candidate"] == 1, "shallow_PF"], "broad_nonrep_candidate"))
        row.update(describe_pf(g.loc[g["strict_candidate"] == 1, "shallow_PF"], "strict_candidate"))
        row["broad_candidate_rate_pct"] = 100.0 * float(g["broad_candidate"].mean())
        row["broad_nonrep_candidate_rate_pct"] = 100.0 * float(g["broad_nonrep_candidate"].mean())
        row["strict_candidate_rate_pct"] = 100.0 * float(g["strict_candidate"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "summary_shallow_pf.csv"
    summary.to_csv(summary_path, index=False)

    contrast_rows = []
    structure = df[(pd.to_numeric(df["corruption_ratio"], errors="coerce") == 0.64) & df["condition"].isin(CONDITION_NAMES)]
    conditions = ["RR", "RU", "UR", "UU"]
    subset_specs = [("all", None), ("broad_candidate", "broad_candidate"), ("broad_nonrep_candidate", "broad_nonrep_candidate")]
    for perturbation, pg in structure.groupby("perturbation", sort=False):
        for subset_name, mask_base in subset_specs:
            for i, ca in enumerate(conditions):
                for cb in conditions[i + 1:]:
                    paired = make_paired_conditions(pg, ca, cb)
                    obs, lo, hi, n_a, n_b, valid_reps = paired_bootstrap_mean_contrast(
                        paired,
                        mask_base=mask_base,
                        seed=args.seed + len(contrast_rows),
                        reps=args.bootstrap_reps,
                    )
                    contrast_rows.append({
                        "perturbation": perturbation,
                        "subset": subset_name,
                        "condition_a": ca,
                        "condition_a_name": CONDITION_NAMES[ca],
                        "condition_b": cb,
                        "condition_b_name": CONDITION_NAMES[cb],
                        "matched_utterance_N": int(len(paired)),
                        "N_a": n_a,
                        "N_b": n_b,
                        "mean_PF_a_minus_b": obs,
                        "paired_bootstrap_95ci_low": lo,
                        "paired_bootstrap_95ci_high": hi,
                        "bootstrap_reps": args.bootstrap_reps,
                        "valid_bootstrap_reps": valid_reps,
                        "bootstrap_design": "matched utterance IDs; condition-specific subset means recomputed within each joint resample",
                    })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts_path = args.output_dir / "pairwise_shallow_pf_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)

    manifest = {
        "analysis": "SHALLOW phonetic fabrication (PF)",
        "source_repository": SHALLOW_REPO,
        "source_commit": SHALLOW_COMMIT,
        "source_files": [SHALLOW_FABRICATIONS_PATH, SHALLOW_AGGREGATION_PATH],
        "jellyfish_version_required": JELLYFISH_VERSION,
        "formula": "PF=(hamming_norm+levenshtein_norm+(1-jaro_winkler))/3 on jellyfish.metaphone strings",
        "interpretation": "Higher PF means greater phonetic fabrication / lower phonetic similarity; no binary SHALLOW PF threshold is imposed.",
        "candidate_thresholds": {"broad_WER_gt": WER_THRESHOLD, "strict_WER_gt": STRICT_WER_THRESHOLD, "Qwen_plausibility_gt": QWEN_THRESHOLD},
        "pairwise_bootstrap": {
            "design": "matched utterance-level bootstrap",
            "unit": "utterance_id",
            "reps": args.bootstrap_reps,
            "subset_handling": "condition-specific candidate masks are reapplied inside each joint bootstrap resample",
            "interpretation": "descriptive 95% bootstrap confidence intervals for paired mean PF contrasts; no null-hypothesis p-values are reported",
        },
        "input": str(args.input),
        "outputs": {"per_utterance": str(per_path), "summary": str(summary_path), "pairwise_contrasts": str(contrasts_path)},
    }
    manifest_path = args.output_dir / "shallow_pf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    display = summary[["condition_name", "perturbation", "N", "all_PF_mean", "all_PF_median", "broad_candidate_rate_pct", "broad_candidate_N", "broad_candidate_PF_mean", "broad_candidate_PF_median", "broad_nonrep_candidate_N", "broad_nonrep_candidate_PF_mean", "strict_candidate_N", "strict_candidate_PF_mean"]]
    print("=== SHALLOW phonetic fabrication (PF) summary ===")
    print(display.to_string(index=False))
    print("\n=== Paired utterance-bootstrap PF contrasts ===")
    print(contrasts.to_string(index=False))
    print(f"\nSaved: {per_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {contrasts_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
