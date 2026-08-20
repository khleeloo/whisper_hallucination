#!/usr/bin/env python3
"""Test whether structured label noise amplifies intrinsic perturbation failures.

Perturbed inputs are per-utterance ``details_*.tsv`` files produced by the stress
evaluation. Clean baselines are the corresponding per-utterance checkpoint CSVs.
The analysis never re-decodes audio.

For metric m, condition c, perturbation p:

    delta[c,p,m] = mean(Y[c,p,m] - Y[c,clean,m])
    DID[c,p,m]   = delta[c,p,m] - delta[Base,p,m]

A positive Base delta together with a bootstrap DID confidence interval entirely
above zero is classified as an amplified intrinsic failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

CONDITIONS = ["Base", "RR", "RU", "UR", "UU"]
METRIC_CANDIDATES = {
    "WER": ["wer", "WER"],
    "Hallucination": ["hallucination_like", "hallucination"],
    "Rep2": ["bigram_rep_count", "2gram_reps", "rep2", "Rep2"],
    "Rep3": ["trigram_rep_count", "3gram_reps", "rep3", "Rep3"],
    "Rep4": ["fourgram_rep_count", "4gram_reps", "rep4", "Rep4"],
    "GroundingGap": ["grounding_gap", "GroundingGap", "grounding"],
    "QwenPlausibility": [
        "normalized_sentence_score_Qwen3-0.6B",
        "qwen_plausibility",
        "Qwen3",
        "norm_plausibility",
    ],
}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_condition(raw: str) -> str:
    value = str(raw).strip().lower()
    for key, out in [("base", "Base"), ("clean", "Base"), ("rr", "RR"), ("ru", "RU"), ("ur", "UR"), ("uu", "UU")]:
        if value == key or value.startswith(key + "_"):
            return out
    raise ValueError(f"Unknown condition: {raw}")


def parse_perturbed_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if stem.startswith("details_"):
        stem = stem[len("details_"):]
    low = stem.lower()
    m = re.search(r"(?:^|_)(base|clean|rr(?:_64pct)?|ru(?:_64pct)?|ur(?:_64pct)?|uu(?:_64pct)?)_(none|full_noise|onset_noise|reverb|silence|leading_silence|speech_band_noise)(.*)$", low)
    if not m:
        raise ValueError(f"Cannot parse condition/perturbation from {path.name}")
    condition = canonical_condition(m.group(1))
    perturbation = m.group(2) + m.group(3)
    return condition, perturbation


def find_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for metric, candidates in METRIC_CANDIDATES.items():
        for col in candidates:
            if col in df.columns:
                found[metric] = col
                break
    return found


def frame_from_df(df: pd.DataFrame, condition: str, perturbation: str, source_file: str) -> pd.DataFrame:
    if "reference" not in df.columns:
        raise ValueError(f"{source_file} has no 'reference' column")
    metric_cols = find_metric_columns(df)
    if not metric_cols:
        raise ValueError(f"No recognized metrics in {source_file}; columns={list(df.columns)}")

    out = pd.DataFrame(index=np.arange(len(df)))
    out["condition"] = condition
    out["perturbation"] = perturbation
    out["source_file"] = source_file
    out["source_row"] = np.arange(len(df), dtype=int)
    out["reference"] = df["reference"].astype(str)
    out["reference_norm"] = df["reference"].map(normalize_text)
    for metric, col in metric_cols.items():
        out[metric] = pd.to_numeric(df[col], errors="coerce")
    return out


def load_perturbed(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seen = set()
    for path_str in sorted(set(paths)):
        path = Path(path_str)
        condition, perturbation = parse_perturbed_filename(path)
        key = (condition, perturbation)
        if key in seen:
            raise ValueError(
                f"Duplicate perturbed condition/tag {key}. Do not mix stress outputs from different checkpoints."
            )
        seen.add(key)
        df = pd.read_csv(path, sep="\t")
        frames.append(frame_from_df(df, condition, perturbation, str(path)))
    if not frames:
        raise ValueError("No perturbed TSVs were loaded")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.sort_values(["condition", "perturbation", "reference_norm"], kind="mergesort")
    data["reference_occurrence"] = data.groupby(
        ["condition", "perturbation", "reference_norm"], sort=False
    ).cumcount()
    return data


def parse_clean_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--clean must be CONDITION=PATH_OR_GLOB, got: {spec}")
    cond, pattern = spec.split("=", 1)
    return canonical_condition(cond), pattern


def load_clean(specs: Sequence[str]) -> pd.DataFrame:
    per_condition: Dict[str, List[pd.DataFrame]] = {c: [] for c in CONDITIONS}
    for spec in specs:
        condition, pattern = parse_clean_spec(spec)
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No clean CSV matched {condition}={pattern}")
        for path_str in matches:
            path = Path(path_str)
            df = pd.read_csv(path)
            per_condition[condition].append(frame_from_df(df, condition, "none", str(path)))

    frames: List[pd.DataFrame] = []
    for condition, parts in per_condition.items():
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True, sort=False)
        merged = merged.sort_values("reference_norm", kind="mergesort")
        merged["reference_occurrence"] = merged.groupby("reference_norm", sort=False).cumcount()
        frames.append(merged)
    if not frames:
        raise ValueError("No clean CSVs were loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def pair_with_clean(perturbed: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    present_conditions = sorted(set(perturbed["condition"]))
    missing_conditions = [c for c in present_conditions if c not in set(clean["condition"])]
    if missing_conditions:
        raise ValueError(f"Missing clean baseline for conditions: {missing_conditions}")

    shared_metrics = [
        m for m in METRIC_CANDIDATES
        if m in perturbed.columns and m in clean.columns
    ]
    if not shared_metrics:
        raise ValueError("Clean CSVs and perturbed TSVs have no shared recognized metrics")

    key = ["condition", "reference_norm", "reference_occurrence"]
    clean_small = clean[key + shared_metrics].copy()
    if clean_small.duplicated(key).any():
        examples = clean_small.loc[clean_small.duplicated(key, keep=False), key].head().to_dict("records")
        raise ValueError(f"Clean pairing keys are not unique; examples={examples}")
    clean_small = clean_small.rename(columns={m: f"{m}_clean" for m in shared_metrics})

    paired = perturbed.merge(clean_small, on=key, how="left", validate="many_to_one")
    sentinel = f"{shared_metrics[0]}_clean"
    missing = int(paired[sentinel].isna().sum())
    if missing:
        print(
            f"WARNING: {missing:,}/{len(paired):,} perturbed rows could not be paired to clean rows. "
            "Dropping unpaired rows. Verify that clean and stress evaluations use the same test split."
        )
        paired = paired.dropna(subset=[sentinel])
    for metric in shared_metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_clean"]
    print(f"Shared metrics: {shared_metrics}")
    return paired


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(x[rng.integers(0, x.size, size=x.size)])
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def bootstrap_did_ci(noisy: np.ndarray, base: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    a = np.asarray(noisy, dtype=float)
    b = np.asarray(base, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals[i] = np.mean(a[rng.integers(0, len(a), size=len(a))]) - np.mean(
            b[rng.integers(0, len(b), size=len(b))]
        )
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def summarize_deltas(paired: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    metrics = [m for m in METRIC_CANDIDATES if f"delta_{m}" in paired.columns]
    rows = []
    for (condition, perturbation), group in paired.groupby(["condition", "perturbation"], sort=True):
        for metric in metrics:
            vals = pd.to_numeric(group[f"delta_{metric}"], errors="coerce").dropna().to_numpy(float)
            if not len(vals):
                continue
            lo, hi = bootstrap_mean_ci(vals, n_boot, seed + len(rows))
            rows.append({"condition": condition, "perturbation": perturbation, "metric": metric,
                         "N": len(vals), "delta_mean": float(np.mean(vals)),
                         "delta_ci_low": lo, "delta_ci_high": hi})
    return pd.DataFrame(rows)


def classify_effect(base_delta: float, noisy_delta: float, lo: float, hi: float, eps: float) -> str:
    if abs(base_delta) <= eps:
        if noisy_delta > eps:
            return "condition-specific increase"
        if noisy_delta < -eps:
            return "condition-specific decrease"
        return "no material response"
    if base_delta > eps:
        if lo > 0:
            return "amplified intrinsic failure"
        if hi < 0:
            return "attenuated intrinsic failure"
        return "intrinsic response reproduced"
    return "base response decreases metric"


def difference_in_differences(paired: pd.DataFrame, summary: pd.DataFrame,
                              n_boot: int, seed: int, eps: float) -> pd.DataFrame:
    rows = []
    metrics = sorted(summary["metric"].unique())
    perturbations = sorted(summary["perturbation"].unique())
    for condition in [c for c in CONDITIONS if c != "Base" and c in set(paired["condition"])]:
        for perturbation in perturbations:
            for metric in metrics:
                col = f"delta_{metric}"
                base = pd.to_numeric(paired.loc[(paired.condition == "Base") & (paired.perturbation == perturbation), col], errors="coerce").dropna().to_numpy(float)
                noisy = pd.to_numeric(paired.loc[(paired.condition == condition) & (paired.perturbation == perturbation), col], errors="coerce").dropna().to_numpy(float)
                if not len(base) or not len(noisy):
                    continue
                bmean, nmean = float(np.mean(base)), float(np.mean(noisy))
                did = nmean - bmean
                lo, hi = bootstrap_did_ci(noisy, base, n_boot, seed + len(rows))
                ratio = nmean / bmean if abs(bmean) > eps and np.sign(bmean) == np.sign(nmean) else np.nan
                rows.append({"condition": condition, "perturbation": perturbation, "metric": metric,
                             "base_delta": bmean, "noisy_delta": nmean,
                             "difference_in_differences": did, "did_ci_low": lo, "did_ci_high": hi,
                             "amplification_ratio": ratio,
                             "interpretation": classify_effect(bmean, nmean, lo, hi, eps)})
    return pd.DataFrame(rows)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else np.nan


def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 2 or np.var(x) == 0:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[0]), float(coef[1]), (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def signature_similarity(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in sorted(summary.metric.unique()):
        base = summary[(summary.condition == "Base") & (summary.metric == metric)][["perturbation", "delta_mean"]].rename(columns={"delta_mean": "base_delta"})
        for condition in [c for c in CONDITIONS if c != "Base"]:
            other = summary[(summary.condition == condition) & (summary.metric == metric)][["perturbation", "delta_mean"]].rename(columns={"delta_mean": "condition_delta"})
            joined = base.merge(other, on="perturbation", how="inner")
            if len(joined) < 2:
                continue
            x, y = joined.base_delta.to_numpy(float), joined.condition_delta.to_numpy(float)
            intercept, slope, r2 = ols(x, y)
            rows.append({"condition": condition, "metric": metric,
                         "n_perturbation_points": len(joined), "spearman": spearman(x, y),
                         "cosine_similarity": cosine(x, y), "ols_intercept": intercept,
                         "amplification_slope_beta": slope, "ols_r2": r2})
    return pd.DataFrame(rows)


def collect_globs(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(set(paths))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbed_glob", action="append", default=[], help="Glob for perturbed details_*.tsv; repeatable")
    p.add_argument("--details_glob", action="append", default=[], help="Backward-compatible alias for --perturbed_glob")
    p.add_argument("--clean", action="append", default=[], help="Clean CSV mapping CONDITION=PATH_OR_GLOB; repeatable")
    p.add_argument("--output_dir", default="results/failure_amplification")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zero_epsilon", type=float, default=1e-6)
    args = p.parse_args()

    perturbed_paths = collect_globs(args.perturbed_glob + args.details_glob)
    if not perturbed_paths:
        raise FileNotFoundError("No perturbed details TSVs matched the supplied globs")
    if not args.clean:
        raise ValueError("At least one --clean CONDITION=PATH_OR_GLOB is required")

    perturbed = load_perturbed(perturbed_paths)
    clean = load_clean(args.clean)
    print(f"Loaded perturbed rows: {len(perturbed):,} from {len(perturbed_paths)} files")
    print(f"Loaded clean rows: {len(clean):,}")
    print("Perturbed conditions:", sorted(perturbed.condition.unique()))
    print("Clean conditions:", sorted(clean.condition.unique()))

    paired = pair_with_clean(perturbed, clean)
    summary = summarize_deltas(paired, args.bootstrap, args.seed)
    did = difference_in_differences(paired, summary, args.bootstrap, args.seed, args.zero_epsilon)
    similarity = signature_similarity(summary)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "paired_clean_to_perturbed_deltas.csv", index=False)
    summary.to_csv(out / "perturbation_delta_summary.csv", index=False)
    did.to_csv(out / "difference_in_differences.csv", index=False)
    similarity.to_csv(out / "signature_similarity.csv", index=False)

    counts = did.interpretation.value_counts().to_dict() if not did.empty else {}
    payload = {
        "decision_rule": {
            "amplified": "Base delta > 0 and bootstrap 95% CI for DID entirely > 0",
            "reproduced": "Base delta > 0 and bootstrap 95% CI for DID includes 0",
            "attenuated": "Base delta > 0 and bootstrap 95% CI for DID entirely < 0",
            "condition_specific": "Base delta approximately 0 while noisy delta is non-zero",
        },
        "interpretation_counts": {str(k): int(v) for k, v in counts.items()},
        "n_paired_rows": int(len(paired)),
        "n_similarity_rows": int(len(similarity)),
    }
    (out / "hypothesis_check_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Failure amplification hypothesis check ===")
    print(f"Paired perturbed rows: {len(paired):,}")
    if not did.empty:
        print("\nInterpretation counts:")
        print(did.interpretation.value_counts().to_string())
        amp = did[did.interpretation == "amplified intrinsic failure"]
        print("\nStrongest amplification candidates:")
        if amp.empty:
            print("  none under the pre-specified rule")
        else:
            print(amp.sort_values("difference_in_differences", ascending=False).head(20).to_string(index=False))
    if not similarity.empty:
        print("\nSignature similarity / amplification slopes:")
        print(similarity.sort_values(["metric", "condition"]).to_string(index=False))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
