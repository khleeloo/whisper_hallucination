#!/usr/bin/env python3
"""Test whether structured label noise amplifies intrinsic perturbation failures.

Perturbed inputs are legacy/new ``details_*.tsv`` stress outputs. Clean baselines
are per-utterance checkpoint CSVs. New stress files should contain an utterance ID;
legacy stress files do not, so their IDs are reconstructed from the exact test.tsv
order used by evaluate_dual_metric.py. WAcc is intentionally ignored everywhere.

For metric m, condition c, perturbation p:
    delta[c,p,m] = mean(Y[c,p,m] - Y[c,clean,m])
    DID[c,p,m]   = delta[c,p,m] - delta[Base,p,m]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

CONDITIONS = ["Base", "RR", "RU", "UR", "UU"]
CV_RE = re.compile(r"(?:common_voice_en_\d+)", re.IGNORECASE)

# WAcc deliberately excluded. Historical files may contain it, but it is not read.
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
    m = re.search(
        r"(?:^|_)(base|clean|rr(?:_64pct)?|ru(?:_64pct)?|ur(?:_64pct)?|uu(?:_64pct)?)_"
        r"(full_noise|onset_noise|reverb|silence|leading_silence|speech_band_noise)(.*)$",
        low,
    )
    if not m:
        raise ValueError(f"Cannot parse condition/perturbation from {path.name}")
    return canonical_condition(m.group(1)), m.group(2) + m.group(3)


def find_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for metric, candidates in METRIC_CANDIDATES.items():
        for col in candidates:
            if col in df.columns:
                found[metric] = col
                break
    return found


def extract_cv_id(value: object) -> str | None:
    m = CV_RE.search(str(value))
    return m.group(0).lower() if m else None


def infer_utterance_ids(df: pd.DataFrame, source_file: str) -> pd.Series | None:
    """Return IDs from an explicit/id-like column, or None for legacy stress TSVs."""
    preferred = [
        "utterance_id", "audio_id", "clip_id", "path", "audio_path",
        "filename", "file", "model_name",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    for col in ordered:
        sample = df[col].astype(str)
        # Avoid pandas regex-group warnings by using a non-capturing regex string.
        rate = sample.str.contains(r"(?:common_voice_en_\d+)", regex=True, na=False).mean()
        if rate >= 0.95:
            ids = sample.map(extract_cv_id)
            if ids.notna().all():
                print(f"ID source for {Path(source_file).name}: column '{col}'")
                return ids.astype(str)
    return None


def build_test_manifest(test_tsv: str, clips_dir: str, max_samples: int | None) -> pd.DataFrame:
    """Reproduce evaluate_dual_metric.load_test_data ordering exactly."""
    rows = []
    with open(test_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            audio_path = os.path.join(clips_dir, row["path"])
            if not os.path.exists(audio_path):
                continue
            uid = extract_cv_id(row["path"])
            if uid is None:
                raise ValueError(f"Cannot recover Common Voice ID from test.tsv path: {row['path']}")
            rows.append({
                "utterance_id": uid,
                "reference_manifest": row["sentence"],
                "audio_path_manifest": audio_path,
            })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No usable rows reconstructed from {test_tsv}")
    return manifest


def frame_from_df(
    df: pd.DataFrame,
    condition: str,
    perturbation: str,
    source_file: str,
    fallback_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if "reference" not in df.columns:
        raise ValueError(f"{source_file} has no 'reference' column")
    metric_cols = find_metric_columns(df)
    if not metric_cols:
        raise ValueError(f"No recognized metrics in {source_file}; columns={list(df.columns)}")

    ids = infer_utterance_ids(df, source_file)
    if ids is None:
        if fallback_manifest is None:
            raise ValueError(
                f"Could not find utterance IDs in {source_file} and no test manifest fallback was supplied. "
                f"Columns={list(df.columns)}"
            )
        if len(df) != len(fallback_manifest):
            raise ValueError(
                f"Legacy stress file {source_file} has {len(df)} rows but reconstructed test manifest "
                f"has {len(fallback_manifest)} rows. Check --stress_max_samples and test.tsv provenance."
            )
        # Legacy stress files were written in the exact order returned by load_test_data.
        ids = fallback_manifest["utterance_id"].reset_index(drop=True)
        stress_ref = df["reference"].map(normalize_text).reset_index(drop=True)
        manifest_ref = fallback_manifest["reference_manifest"].map(normalize_text).reset_index(drop=True)
        mismatch = stress_ref != manifest_ref
        if mismatch.any():
            idx = np.flatnonzero(mismatch.to_numpy())[:5]
            examples = [
                {
                    "row": int(i),
                    "stress_reference": df.iloc[i]["reference"],
                    "manifest_reference": fallback_manifest.iloc[i]["reference_manifest"],
                }
                for i in idx
            ]
            raise ValueError(
                f"Legacy stress row order does not match test.tsv for {source_file}; "
                f"{int(mismatch.sum())}/{len(df)} reference mismatches. Examples={examples}"
            )
        print(f"ID source for {Path(source_file).name}: reconstructed from test.tsv row order")

    out = pd.DataFrame(index=np.arange(len(df)))
    out["condition"] = condition
    out["perturbation"] = perturbation
    out["source_file"] = source_file
    out["source_row"] = np.arange(len(df), dtype=int)
    out["utterance_id"] = ids.astype(str).str.lower().to_numpy()
    out["reference"] = df["reference"].astype(str).to_numpy()
    out["reference_norm"] = df["reference"].map(normalize_text).to_numpy()
    for metric, col in metric_cols.items():
        out[metric] = pd.to_numeric(df[col], errors="coerce").to_numpy()
    return out


def load_perturbed(paths: Sequence[str], manifest: pd.DataFrame | None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seen = set()
    for path_str in sorted(set(paths)):
        path = Path(path_str)
        condition, perturbation = parse_perturbed_filename(path)
        key = (condition, perturbation)
        if key in seen:
            raise ValueError(f"Duplicate perturbed condition/tag {key}; do not mix checkpoints")
        seen.add(key)
        df = pd.read_csv(path, sep="\t")
        frames.append(frame_from_df(df, condition, perturbation, str(path), fallback_manifest=manifest))
    if not frames:
        raise ValueError("No perturbed TSVs were loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_clean_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--clean must be CONDITION=PATH_OR_GLOB, got: {spec}")
    cond, pattern = spec.split("=", 1)
    return canonical_condition(cond), pattern


def load_clean(specs: Sequence[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        condition, pattern = parse_clean_spec(spec)
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No clean CSV matched {condition}={pattern}")
        parts = []
        for path_str in matches:
            df = pd.read_csv(path_str)
            parts.append(frame_from_df(df, condition, "none", path_str))
        merged = pd.concat(parts, ignore_index=True, sort=False)
        if merged["utterance_id"].duplicated().any():
            dups = merged.loc[merged.utterance_id.duplicated(keep=False), "utterance_id"].head().tolist()
            raise ValueError(f"Duplicate clean utterance IDs for {condition}: {dups}")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True, sort=False)


def pair_with_clean(perturbed: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    shared_metrics = [m for m in METRIC_CANDIDATES if m in perturbed.columns and m in clean.columns]
    if not shared_metrics:
        raise ValueError("Clean CSVs and perturbed TSVs have no shared recognized metrics")
    key = ["condition", "utterance_id"]
    clean_small = clean[key + ["reference_norm"] + shared_metrics].copy()
    clean_small = clean_small.rename(
        columns={"reference_norm": "reference_norm_clean", **{m: f"{m}_clean" for m in shared_metrics}}
    )
    paired = perturbed.merge(clean_small, on=key, how="left", validate="many_to_one")
    sentinel = f"{shared_metrics[0]}_clean"
    missing = paired[sentinel].isna()
    if missing.any():
        examples = paired.loc[missing, ["condition", "utterance_id", "source_file"]].head().to_dict("records")
        raise ValueError(
            f"{int(missing.sum()):,}/{len(paired):,} perturbed rows could not be paired by utterance ID. "
            f"Examples={examples}"
        )
    ref_mismatch = paired["reference_norm"] != paired["reference_norm_clean"]
    if ref_mismatch.any():
        examples = paired.loc[ref_mismatch, ["condition", "utterance_id", "reference", "reference_norm_clean"]].head().to_dict("records")
        raise ValueError(
            f"{int(ref_mismatch.sum()):,}/{len(paired):,} paired rows have mismatching references. Examples={examples}"
        )
    for metric in shared_metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_clean"]
    print(f"Paired all {len(paired):,} perturbed rows by Common Voice utterance ID")
    print(f"Shared metrics: {shared_metrics}")
    return paired


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = np.mean(x[rng.integers(0, len(x), len(x))])
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def bootstrap_did_ci(noisy: np.ndarray, base: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    a = np.asarray(noisy, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(base, dtype=float); b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = np.mean(a[rng.integers(0, len(a), len(a))]) - np.mean(b[rng.integers(0, len(b), len(b))])
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
        if noisy_delta > eps: return "condition-specific increase"
        if noisy_delta < -eps: return "condition-specific decrease"
        return "no material response"
    if base_delta > eps:
        if lo > 0: return "amplified intrinsic failure"
        if hi < 0: return "attenuated intrinsic failure"
        return "intrinsic response reproduced"
    return "base response decreases metric"


def difference_in_differences(paired: pd.DataFrame, summary: pd.DataFrame, n_boot: int, seed: int, eps: float) -> pd.DataFrame:
    rows = []
    for condition in [c for c in CONDITIONS if c != "Base" and c in set(paired.condition)]:
        for perturbation in sorted(summary.perturbation.unique()):
            for metric in sorted(summary.metric.unique()):
                col = f"delta_{metric}"
                if col not in paired.columns:
                    continue
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
    if len(x) < 2 or len(y) < 2: return np.nan
    rx = pd.Series(x).rank().to_numpy(float); ry = pd.Series(y).rank().to_numpy(float)
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else np.nan


def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 2 or np.var(x) == 0: return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[0]), float(coef[1]), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def signature_similarity(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in sorted(summary.metric.unique()):
        base = summary[(summary.condition == "Base") & (summary.metric == metric)][["perturbation", "delta_mean"]].rename(columns={"delta_mean": "base_delta"})
        for condition in [c for c in CONDITIONS if c != "Base"]:
            other = summary[(summary.condition == condition) & (summary.metric == metric)][["perturbation", "delta_mean"]].rename(columns={"delta_mean": "condition_delta"})
            joined = base.merge(other, on="perturbation", how="inner")
            if len(joined) < 2: continue
            x, y = joined.base_delta.to_numpy(float), joined.condition_delta.to_numpy(float)
            intercept, slope, r2 = ols(x, y)
            rows.append({"condition": condition, "metric": metric,
                         "n_perturbation_points": len(joined), "spearman": spearman(x, y),
                         "cosine_similarity": cosine(x, y), "ols_intercept": intercept,
                         "amplification_slope_beta": slope, "ols_r2": r2})
    return pd.DataFrame(rows)


def collect_globs(patterns: Sequence[str]) -> List[str]:
    paths = []
    for pattern in patterns: paths.extend(glob.glob(pattern))
    return sorted(set(paths))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbed_glob", action="append", default=[])
    p.add_argument("--details_glob", action="append", default=[], help="Alias for --perturbed_glob")
    p.add_argument("--clean", action="append", default=[], help="CONDITION=PATH_OR_GLOB")
    p.add_argument("--test_tsv", default=None, help="Required for legacy stress TSVs without utterance IDs")
    p.add_argument("--clips_dir", default=None, help="Required with --test_tsv")
    p.add_argument("--stress_max_samples", type=int, default=None, help="Must match stress evaluation --max_samples")
    p.add_argument("--output_dir", default="results/failure_amplification")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zero_epsilon", type=float, default=1e-6)
    args = p.parse_args()

    perturbed_paths = collect_globs(args.perturbed_glob + args.details_glob)
    if not perturbed_paths: raise FileNotFoundError("No perturbed details TSVs matched")
    if not args.clean: raise ValueError("At least one --clean CONDITION=PATH_OR_GLOB is required")

    manifest = None
    if args.test_tsv or args.clips_dir:
        if not args.test_tsv or not args.clips_dir:
            raise ValueError("--test_tsv and --clips_dir must be provided together")
        manifest = build_test_manifest(args.test_tsv, args.clips_dir, args.stress_max_samples)
        print(f"Reconstructed stress manifest: {len(manifest):,} rows")

    perturbed = load_perturbed(perturbed_paths, manifest)
    clean = load_clean(args.clean)
    print(f"Loaded perturbed rows: {len(perturbed):,} from {len(perturbed_paths)} files")
    print(f"Loaded clean rows: {len(clean):,}")

    paired = pair_with_clean(perturbed, clean)
    summary = summarize_deltas(paired, args.bootstrap, args.seed)
    did = difference_in_differences(paired, summary, args.bootstrap, args.seed, args.zero_epsilon)
    similarity = signature_similarity(summary)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "paired_clean_to_perturbed_deltas.csv", index=False)
    summary.to_csv(out / "perturbation_delta_summary.csv", index=False)
    did.to_csv(out / "difference_in_differences.csv", index=False)
    similarity.to_csv(out / "signature_similarity.csv", index=False)
    payload = {
        "decision_rule": {
            "amplified": "Base delta > 0 and bootstrap 95% CI for DID entirely > 0",
            "reproduced": "Base delta > 0 and bootstrap 95% CI for DID includes 0",
            "attenuated": "Base delta > 0 and bootstrap 95% CI for DID entirely < 0",
        },
        "interpretation_counts": {str(k): int(v) for k, v in did.interpretation.value_counts().to_dict().items()} if not did.empty else {},
        "n_paired_rows": int(len(paired)),
        "wacc_used": False,
    }
    (out / "hypothesis_check_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Failure amplification hypothesis check ===")
    print(f"Paired perturbed rows: {len(paired):,}")
    if not did.empty:
        print(did.interpretation.value_counts().to_string())
    if not similarity.empty:
        print("\nSignature similarity / amplification slopes:")
        print(similarity.sort_values(["metric", "condition"]).to_string(index=False))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
