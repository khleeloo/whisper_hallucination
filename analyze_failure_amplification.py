#!/usr/bin/env python3
"""Test whether structured label noise amplifies intrinsic perturbation failures.

This analysis treats the clean Base model's response to acoustic perturbation as the
intrinsic Whisper failure signature. For each noisy condition (RR/RU/UR/UU), it asks:

1. Does the perturbation response point in the same direction as Base?
2. Is the response larger than Base (amplification) rather than merely different?
3. Are any effects condition-specific rather than amplified intrinsic behavior?

The script operates on per-utterance detail TSVs produced by the evaluation pipeline.
It never changes model outputs and does not require re-decoding.

Core quantities for metric m, condition c, perturbation p:

    delta[c,p,m] = mean(Y[c,p,m] - Y[c,clean,m])

and the difference-in-differences relative to Base:

    DID[c,p,m] = delta[c,p,m] - delta[Base,p,m]

Interpretation for failure-oriented metrics (WER, hallucination_like, repetition,
grounding_gap where larger means worse):

    Base delta > 0 and DID > 0  -> amplified intrinsic failure
    Base delta > 0 and DID ~= 0 -> reproduced intrinsic failure
    Base delta > 0 and DID < 0  -> attenuated intrinsic failure
    Base delta ~= 0 and noisy delta > 0 -> condition-specific / induced failure

Example:
    python analyze_failure_amplification.py \
      --details_glob '/scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_eval_64pct/details_*.tsv' \
      --output_dir results/failure_amplification

You may pass multiple --details_glob arguments. Clean files (perturbation=none) must
be present for every condition being analyzed. Pairing is verified by normalized
reference text and within-file row occurrence, so duplicated references remain paired.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


CONDITIONS = ["Base", "RR", "RU", "UR", "UU"]
PERTURBATION_PREFIXES = [
    "none",
    "full_noise",
    "onset_noise",
    "reverb",
    "silence",
    "leading_silence",
    "speech_band_noise",
]

# Candidate source columns -> canonical metric names. Only metrics present in the
# supplied TSVs are analyzed.
METRIC_CANDIDATES = {
    "WER": ["wer", "WER"],
    "Hallucination": ["hallucination_like", "hallucination"],
    "Rep2": ["2gram_reps", "rep2", "Rep2"],
    "Rep3": ["3gram_reps", "rep3", "Rep3"],
    "Rep4": ["4gram_reps", "rep4", "Rep4"],
    "GroundingGap": ["grounding_gap", "GroundingGap", "grounding"],
    "QwenPlausibility": ["qwen_plausibility", "Qwen3", "norm_plausibility"],
}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_condition(raw: str) -> str:
    value = str(raw).strip().lower()
    mapping = {
        "base": "Base",
        "clean": "Base",
        "rr": "RR",
        "ru": "RU",
        "ur": "UR",
        "uu": "UU",
    }
    for key, canonical in mapping.items():
        if value == key or value.startswith(key + "_"):
            return canonical
    return str(raw)


def parse_detail_filename(path: Path) -> Tuple[str, str]:
    """Parse details_<condition>_<perturbation>.tsv-style names.

    The parser deliberately preserves the full perturbation tag (including amplitude
    and duration), because different severity levels are separate response points.
    """
    stem = path.stem
    if stem.startswith("details_"):
        stem = stem[len("details_"):]

    lower = stem.lower()
    condition = None
    remainder = None
    for candidate in ["base", "clean", "rr", "ru", "ur", "uu"]:
        prefix = candidate + "_"
        if lower == candidate:
            condition = canonical_condition(candidate)
            remainder = "none"
            break
        if lower.startswith(prefix):
            condition = canonical_condition(candidate)
            remainder = stem[len(prefix):]
            break

    if condition is None:
        # Fallback for names with additional run prefixes. Find the first condition
        # token followed by a recognized perturbation token.
        m = re.search(
            r"(?:^|_)(base|clean|rr|ru|ur|uu)_(none|full_noise|onset_noise|reverb|silence|leading_silence|speech_band_noise)(.*)$",
            lower,
        )
        if not m:
            raise ValueError(f"Cannot parse condition/perturbation from {path.name}")
        condition = canonical_condition(m.group(1))
        remainder = m.group(2) + m.group(3)

    perturbation = str(remainder or "none")
    if perturbation in {"clean", "baseline"}:
        perturbation = "none"
    return condition, perturbation


def find_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for metric, candidates in METRIC_CANDIDATES.items():
        for col in candidates:
            if col in df.columns:
                found[metric] = col
                break
    return found


def load_details(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    metric_union: Dict[str, str] = {}

    for path_str in sorted(set(paths)):
        path = Path(path_str)
        condition, perturbation = parse_detail_filename(path)
        df = pd.read_csv(path, sep="\t")
        if "reference" not in df.columns:
            raise ValueError(f"{path} has no 'reference' column")

        metric_cols = find_metric_columns(df)
        if not metric_cols:
            raise ValueError(f"No recognized metrics in {path}; columns={list(df.columns)}")

        part = pd.DataFrame(index=np.arange(len(df)))
        part["condition"] = condition
        part["perturbation"] = perturbation
        part["source_file"] = str(path)
        part["source_row"] = np.arange(len(df), dtype=int)
        part["reference"] = df["reference"].astype(str)
        part["reference_norm"] = df["reference"].map(normalize_text)

        # occurrence index makes duplicated reference strings pairable.
        part["reference_occurrence"] = part.groupby("reference_norm", sort=False).cumcount()
        for metric, source_col in metric_cols.items():
            part[metric] = pd.to_numeric(df[source_col], errors="coerce")
            metric_union[metric] = source_col
        frames.append(part)

    if not frames:
        raise ValueError("No detail TSVs were loaded")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Loaded {len(combined):,} rows from {len(frames)} files")
    print("Conditions:", sorted(combined["condition"].unique()))
    print("Perturbations:", sorted(combined["perturbation"].unique()))
    print("Metrics:", [m for m in METRIC_CANDIDATES if m in combined.columns])
    return combined


def pair_with_clean(data: pd.DataFrame) -> pd.DataFrame:
    key = ["condition", "reference_norm", "reference_occurrence"]
    metrics = [m for m in METRIC_CANDIDATES if m in data.columns]
    clean = data[data["perturbation"] == "none"][key + metrics].copy()
    if clean.empty:
        raise ValueError("No clean (perturbation=none) detail rows found")

    duplicate_keys = clean.duplicated(key, keep=False)
    if duplicate_keys.any():
        examples = clean.loc[duplicate_keys, key].head().to_dict("records")
        raise ValueError(f"Clean pairing keys are not unique; examples={examples}")

    clean = clean.rename(columns={m: f"{m}_clean" for m in metrics})
    pert = data[data["perturbation"] != "none"].copy()
    paired = pert.merge(clean, on=key, how="left", validate="many_to_one")

    missing = paired[f"{metrics[0]}_clean"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing:,}/{len(paired):,} perturbed rows could not be paired to clean rows. "
            "Check that clean and perturbed evaluations use the same test utterances."
        )

    for metric in metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_clean"]
    return paired


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1 or n_boot <= 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    # Chunking avoids allocating an n_boot x n matrix for large evaluations.
    for i in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        means[i] = float(np.mean(x[idx]))
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def bootstrap_did_ci(
    noisy_delta: np.ndarray,
    base_delta: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    a = np.asarray(noisy_delta, dtype=float)
    b = np.asarray(base_delta, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ai = a[rng.integers(0, a.size, size=a.size)]
        bi = b[rng.integers(0, b.size, size=b.size)]
        values[i] = float(np.mean(ai) - np.mean(bi))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def summarize_deltas(paired: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    metrics = [m for m in METRIC_CANDIDATES if f"delta_{m}" in paired.columns]
    rows = []
    for (condition, perturbation), group in paired.groupby(["condition", "perturbation"], sort=True):
        for metric in metrics:
            vals = pd.to_numeric(group[f"delta_{metric}"], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed + len(rows))
            rows.append({
                "condition": condition,
                "perturbation": perturbation,
                "metric": metric,
                "N": int(len(vals)),
                "delta_mean": float(np.mean(vals)),
                "delta_ci_low": float(lo),
                "delta_ci_high": float(hi),
            })
    return pd.DataFrame(rows)


def classify_effect(base_delta: float, noisy_delta: float, did_lo: float, did_hi: float, eps: float) -> str:
    if not np.isfinite(base_delta) or not np.isfinite(noisy_delta):
        return "insufficient"
    if abs(base_delta) <= eps:
        if noisy_delta > eps:
            return "condition-specific increase"
        if noisy_delta < -eps:
            return "condition-specific decrease"
        return "no material response"
    if base_delta > eps:
        if did_lo > 0:
            return "amplified intrinsic failure"
        if did_hi < 0:
            return "attenuated intrinsic failure"
        return "intrinsic response reproduced"
    # If the Base perturbation improves the metric, 'amplification' is not a useful
    # failure label. Preserve the sign information without over-interpreting it.
    return "base response decreases metric"


def difference_in_differences(
    paired: pd.DataFrame,
    delta_summary: pd.DataFrame,
    n_boot: int,
    seed: int,
    eps: float,
) -> pd.DataFrame:
    metrics = sorted(delta_summary["metric"].unique())
    perturbations = sorted(delta_summary["perturbation"].unique())
    noisy_conditions = [c for c in CONDITIONS if c != "Base" and c in set(paired["condition"])]
    rows = []

    for condition in noisy_conditions:
        for perturbation in perturbations:
            for metric in metrics:
                base = paired[(paired.condition == "Base") & (paired.perturbation == perturbation)]
                noisy = paired[(paired.condition == condition) & (paired.perturbation == perturbation)]
                col = f"delta_{metric}"
                if col not in paired.columns or base.empty or noisy.empty:
                    continue
                b = pd.to_numeric(base[col], errors="coerce").to_numpy(float)
                n = pd.to_numeric(noisy[col], errors="coerce").to_numpy(float)
                b = b[np.isfinite(b)]
                n = n[np.isfinite(n)]
                if not len(b) or not len(n):
                    continue

                bmean = float(np.mean(b))
                nmean = float(np.mean(n))
                did = nmean - bmean
                lo, hi = bootstrap_did_ci(n, b, n_boot=n_boot, seed=seed + len(rows))
                ratio = np.nan
                if abs(bmean) > eps and np.sign(bmean) == np.sign(nmean):
                    ratio = nmean / bmean
                rows.append({
                    "condition": condition,
                    "perturbation": perturbation,
                    "metric": metric,
                    "base_delta": bmean,
                    "noisy_delta": nmean,
                    "difference_in_differences": did,
                    "did_ci_low": float(lo),
                    "did_ci_high": float(hi),
                    "amplification_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
                    "interpretation": classify_effect(bmean, nmean, lo, hi, eps),
                })
    return pd.DataFrame(rows)


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, implemented without scipy."""
    s = pd.Series(values)
    return s.rank(method="average").to_numpy(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    rx, ry = rank_average(x), rank_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 2 or np.var(x) == 0:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0]), float(coef[1]), float(r2)


def signature_similarity(delta_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each condition's perturbation-response shape against Base per metric."""
    rows = []
    for metric in sorted(delta_summary.metric.unique()):
        base = (
            delta_summary[(delta_summary.condition == "Base") & (delta_summary.metric == metric)]
            [["perturbation", "delta_mean"]]
            .rename(columns={"delta_mean": "base_delta"})
        )
        if base.empty:
            continue
        for condition in [c for c in CONDITIONS if c != "Base"]:
            other = (
                delta_summary[(delta_summary.condition == condition) & (delta_summary.metric == metric)]
                [["perturbation", "delta_mean"]]
                .rename(columns={"delta_mean": "condition_delta"})
            )
            joined = base.merge(other, on="perturbation", how="inner")
            if len(joined) < 2:
                continue
            x = joined.base_delta.to_numpy(float)
            y = joined.condition_delta.to_numpy(float)
            intercept, slope, r2 = ols_slope_intercept(x, y)
            rows.append({
                "condition": condition,
                "metric": metric,
                "n_perturbation_points": int(len(joined)),
                "spearman": spearman(x, y),
                "cosine_similarity": cosine_similarity(x, y),
                "ols_intercept": intercept,
                "amplification_slope_beta": slope,
                "ols_r2": r2,
            })
    return pd.DataFrame(rows)


def collect_paths(globs_: Sequence[str], files: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in globs_:
        paths.extend(glob.glob(pattern))
    paths.extend(files)
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No detail TSV files matched --details_glob/--details")
    return paths


def write_hypothesis_summary(did: pd.DataFrame, similarity: pd.DataFrame, output_path: Path) -> None:
    counts = did["interpretation"].value_counts().to_dict() if not did.empty else {}
    payload = {
        "decision_rule": {
            "support_amplification": "Base delta > 0 and bootstrap 95% CI for DID is entirely > 0",
            "support_reproduction": "Base delta > 0 and bootstrap 95% CI for DID includes 0",
            "support_attenuation": "Base delta > 0 and bootstrap 95% CI for DID is entirely < 0",
            "condition_specific": "Base delta is approximately 0 while noisy delta is materially non-zero",
            "signature_alignment": "Use Spearman/cosine plus slope; beta>1 alone is not evidence if alignment is poor",
        },
        "interpretation_counts": {str(k): int(v) for k, v in counts.items()},
        "n_similarity_rows": int(len(similarity)),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details_glob", action="append", default=[], help="Glob for detail TSVs; repeatable")
    parser.add_argument("--details", action="append", default=[], help="Explicit detail TSV; repeatable")
    parser.add_argument("--output_dir", default="results/failure_amplification")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--zero_epsilon",
        type=float,
        default=1e-6,
        help="Absolute delta treated as approximately zero for qualitative classification",
    )
    args = parser.parse_args()

    paths = collect_paths(args.details_glob, args.details)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_details(paths)
    paired = pair_with_clean(data)
    delta_summary = summarize_deltas(paired, n_boot=args.bootstrap, seed=args.seed)
    did = difference_in_differences(
        paired,
        delta_summary,
        n_boot=args.bootstrap,
        seed=args.seed,
        eps=args.zero_epsilon,
    )
    similarity = signature_similarity(delta_summary)

    paired.to_csv(output_dir / "paired_clean_to_perturbed_deltas.csv", index=False)
    delta_summary.to_csv(output_dir / "perturbation_delta_summary.csv", index=False)
    did.to_csv(output_dir / "difference_in_differences.csv", index=False)
    similarity.to_csv(output_dir / "signature_similarity.csv", index=False)
    write_hypothesis_summary(did, similarity, output_dir / "hypothesis_check_summary.json")

    print("\n=== Failure amplification hypothesis check ===")
    print(f"Paired perturbed rows: {len(paired):,}")
    if not did.empty:
        print("\nInterpretation counts:")
        print(did["interpretation"].value_counts().to_string())
        print("\nStrongest amplification candidates:")
        cols = [
            "condition", "perturbation", "metric", "base_delta", "noisy_delta",
            "difference_in_differences", "did_ci_low", "did_ci_high", "amplification_ratio",
        ]
        amp = did[did.interpretation == "amplified intrinsic failure"]
        if amp.empty:
            print("  none under the pre-specified rule")
        else:
            print(amp[cols].sort_values("difference_in_differences", ascending=False).head(20).to_string(index=False))
    if not similarity.empty:
        print("\nSignature similarity / amplification slopes:")
        print(similarity.sort_values(["metric", "condition"]).to_string(index=False))

    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
