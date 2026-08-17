#!/usr/bin/env python3
"""Experiment 1: repetition-penalty mitigation evaluation.

This module keeps the Experiment 1 workflow explicit:
1. verify repetition_penalty=1.00 against the clean Experiment 0 baseline;
2. select one global penalty on clean DEV only;
3. freeze that penalty;
4. evaluate clean TEST and perturbed TEST separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from evaluate_whisper_validation import load_test_data, normalize_text
from mitigation_experiment import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CLIPS_DIR,
    DEFAULT_CTC_CACHE,
    DEFAULT_GPT2_MODEL,
    DEFAULT_MODEL_CONFIGS,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_PERTURBATIONS,
    DEFAULT_QWEN_MODEL,
    DEFAULT_TEST_TSV,
    DEFAULT_THRESHOLDS_JSON,
    DEFAULT_THRESHOLD_SOURCE_CSV,
    DEFAULT_WAV2VEC2_MODEL,
    OUTPUT_COLUMNS,
    Perturbation,
    _load_whisper_model,
    evaluate_hypotheses,
    load_or_create_frozen_hallucination_thresholds,
    selected_perturbations,
)

CONDITIONS = ["Base", "RR", "RU", "UR", "UU"]
CANDIDATE_PENALTIES = [1.00, 1.05, 1.10, 1.15, 1.20]
KEY_COLUMNS = ["utterance_id", "condition", "perturbation", "reference"]
METRIC_COLUMNS = ["WER", "qwen_plaus", "gpt2_plaus", "rep2", "rep3", "rep4", "hallucination_like", "ctc_nll", "grounding_gap"]
DELTA_METRICS = ["WER", "hallucination_like", "grounding_gap", "rep3", "rep4"]
EXPECTED_CLEAN_WERS = {
    "Base": 0.128629,
    "RR": 0.135792,
    "RU": 0.135806,
    "UR": 0.435395,
    "UU": 0.138380,
}
DEFAULT_RESULTS_DIR = Path("results/mitigation")
DEFAULT_SPLITS_CSV = DEFAULT_RESULTS_DIR / "repetition_penalty_splits.csv"
DEFAULT_REPRO_REPORT = DEFAULT_RESULTS_DIR / "repetition_penalty_1.00_reproduction_report.json"
DEFAULT_DEV_OUTPUTS = DEFAULT_RESULTS_DIR / "repetition_penalty_dev_outputs.csv"
DEFAULT_DEV_GRID = DEFAULT_RESULTS_DIR / "repetition_penalty_dev_grid.csv"
DEFAULT_CONFIG = DEFAULT_RESULTS_DIR / "repetition_penalty_config.json"
DEFAULT_CLEAN_TEST_OUTPUTS = DEFAULT_RESULTS_DIR / "repetition_penalty_clean_test_outputs.csv"
DEFAULT_CLEAN_BEFORE_AFTER = DEFAULT_RESULTS_DIR / "repetition_penalty_before_after.csv"
DEFAULT_CLEAN_SUMMARY = DEFAULT_RESULTS_DIR / "repetition_penalty_summary_by_condition.csv"
DEFAULT_CLEAN_STATS = DEFAULT_RESULTS_DIR / "repetition_penalty_paired_stats.csv"
DEFAULT_PERTURBED_BEFORE_AFTER = DEFAULT_RESULTS_DIR / "repetition_penalty_perturbed_before_after.csv"
DEFAULT_PERTURBED_SUMMARY = DEFAULT_RESULTS_DIR / "repetition_penalty_perturbed_summary.csv"
DEFAULT_PERTURBED_STATS = DEFAULT_RESULTS_DIR / "repetition_penalty_perturbed_paired_stats.csv"
DEFAULT_MECHANISM_CHECK = DEFAULT_RESULTS_DIR / "repetition_penalty_mechanism_check.json"


def stable_split_value(utterance_id: str, seed: int) -> float:
    payload = f"{seed}:{utterance_id}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return int(digest, 16) / float(16 ** 12)


def assign_splits(baseline: pd.DataFrame, dev_fraction: float, seed: int) -> pd.DataFrame:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be between 0 and 1, got {dev_fraction}")
    utterances = sorted(str(value) for value in baseline["utterance_id"].drop_duplicates())
    rows = []
    for utterance_id in utterances:
        split_value = stable_split_value(utterance_id, seed)
        rows.append(
            {
                "utterance_id": utterance_id,
                "split_value": split_value,
                "split": "dev" if split_value < dev_fraction else "test",
                "dev_fraction": dev_fraction,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def load_or_create_splits(
    baseline: pd.DataFrame,
    splits_csv: Path | str = DEFAULT_SPLITS_CSV,
    dev_fraction: float = 0.2,
    seed: int = 1729,
) -> pd.DataFrame:
    path = Path(splits_csv)
    if path.exists():
        splits = pd.read_csv(path)
        required = {"utterance_id", "split", "split_value", "dev_fraction", "seed"}
        missing = required - set(splits.columns)
        if missing:
            raise ValueError(f"Existing split file {path} is missing columns: {sorted(missing)}")
        return splits
    splits = assign_splits(baseline, dev_fraction, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    return splits


def read_clean_baseline(baseline_csv: Path | str = DEFAULT_OUTPUT_CSV) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_csv)
    missing = [column for column in OUTPUT_COLUMNS if column not in baseline.columns]
    if missing:
        raise ValueError(f"Baseline CSV is missing required columns: {missing}")
    clean = baseline[baseline["perturbation"] == "none"].copy()
    if clean.empty:
        raise ValueError("Experiment 0 baseline has no clean perturbation=none rows")
    return clean


def samples_for_utterances(test_tsv: Path | str, clips_dir: Path | str, utterance_ids: Sequence[str]) -> List[Dict[str, str]]:
    wanted = set(str(value) for value in utterance_ids)
    samples = [sample for sample in load_test_data(str(test_tsv), str(clips_dir)) if str(sample.get("utt_id")) in wanted]
    found = {str(sample.get("utt_id")) for sample in samples}
    missing = sorted(wanted - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Could not find {len(missing)} requested utterances in manifest, e.g. {preview}")
    by_id = {str(sample.get("utt_id")): sample for sample in samples}
    return [by_id[str(utterance_id)] for utterance_id in utterance_ids]


def deterministic_decode_seed(seed: Optional[int], condition: str, perturbation: str) -> Optional[int]:
    if seed is None:
        return None
    payload = f"{seed}:{condition}:{perturbation}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def set_decode_seed(seed: Optional[int], condition: str, perturbation: str) -> None:
    resolved = deterministic_decode_seed(seed, condition, perturbation)
    if resolved is None:
        return
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def decode_and_score(
    *,
    utterance_ids: Sequence[str],
    conditions: Sequence[str],
    perturbations: Sequence[Perturbation],
    repetition_penalty: float,
    output_csv: Optional[Path | str] = None,
    test_tsv: Path | str = DEFAULT_TEST_TSV,
    clips_dir: Path | str = DEFAULT_CLIPS_DIR,
    batch_size: int = 8,
    lm_batch_size: int = 8,
    base_model_name: str = DEFAULT_BASE_MODEL,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    gpt2_model: str = DEFAULT_GPT2_MODEL,
    wav2vec2_model: str = DEFAULT_WAV2VEC2_MODEL,
    ctc_cache_path: Path | str = DEFAULT_CTC_CACHE,
    thresholds_json: Path | str = DEFAULT_THRESHOLDS_JSON,
    threshold_source_csv: Path | str = DEFAULT_THRESHOLD_SOURCE_CSV,
    score_lms: bool = True,
    score_ctc: bool = True,
    perturbation_seed: Optional[int] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ordered_ids = [str(value) for value in utterance_ids]
    samples = samples_for_utterances(test_tsv, clips_dir, ordered_ids)
    audio_paths = [sample["audio_path"] for sample in samples]
    references = [sample["reference"] for sample in samples]
    thresholds = load_or_create_frozen_hallucination_thresholds(thresholds_json, threshold_source_csv)
    frames = []
    from evaluate_dual_metric import transcribe_batch as transcribe_perturbed_batch

    for condition in conditions:
        if condition not in DEFAULT_MODEL_CONFIGS:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {sorted(DEFAULT_MODEL_CONFIGS)}")
        config = DEFAULT_MODEL_CONFIGS[condition]
        model, base_model, processor = _load_whisper_model(Path(config["model_dir"]), base_model_name, device)
        try:
            for perturbation in perturbations:
                set_decode_seed(perturbation_seed, condition, perturbation.label)
                hypotheses = transcribe_perturbed_batch(
                    model,
                    processor,
                    audio_paths,
                    perturb_type=perturbation.perturb_type,
                    perturb_amplitude=perturbation.amplitude,
                    perturb_duration=perturbation.duration,
                    device=device,
                    batch_size=batch_size,
                    repetition_penalty=repetition_penalty,
                )
                set_decode_seed(perturbation_seed, condition, perturbation.label)
                scored = evaluate_hypotheses(
                    ordered_ids,
                    [condition] * len(samples),
                    [perturbation.label] * len(samples),
                    references,
                    hypotheses,
                    audio_paths,
                    device=device,
                    qwen_model=qwen_model,
                    gpt2_model=gpt2_model,
                    lm_batch_size=lm_batch_size,
                    wav2vec2_model=wav2vec2_model,
                    ctc_cache_path=ctc_cache_path,
                    score_lms=score_lms,
                    score_ctc=score_ctc,
                    hallucination_thresholds=thresholds,
                )
                scored["repetition_penalty"] = repetition_penalty
                scored["perturbation_type"] = perturbation.perturb_type
                scored["amplitude"] = perturbation.amplitude
                scored["duration"] = perturbation.duration
                frames.append(scored)
        finally:
            model.cpu()
            del model, base_model, processor
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def verify_reproduction(
    generated: pd.DataFrame,
    baseline: pd.DataFrame,
    expected_wers: Optional[Dict[str, float]] = None,
    tolerance: float = 0.02,
) -> Dict[str, object]:
    expected_wers = expected_wers or EXPECTED_CLEAN_WERS
    generated_clean = generated[generated["perturbation"] == "none"].copy()
    baseline_clean = baseline[baseline["perturbation"] == "none"].copy()
    merged = baseline_clean.merge(
        generated_clean,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_baseline", "_generated"),
        indicator=True,
    )
    missing_keys = merged[merged["_merge"] != "both"][KEY_COLUMNS + ["_merge"]].to_dict("records")
    matched = merged[merged["_merge"] == "both"].copy()
    comparisons = []
    for condition in CONDITIONS:
        condition_rows = matched[matched["condition"] == condition]
        if condition_rows.empty:
            comparisons.append({"condition": condition, "missing": True, "passes_tolerance": False})
            continue
        actual_mean = float(condition_rows["WER_generated"].astype(float).mean())
        baseline_mean = float(condition_rows["WER_baseline"].astype(float).mean())
        expected_mean = float(expected_wers[condition])
        exact_hyp_match = condition_rows["hypothesis_baseline"].fillna("").astype(str) == condition_rows[
            "hypothesis_generated"
        ].fillna("").astype(str)
        norm_hyp_match = condition_rows["hypothesis_baseline"].fillna("").map(normalize_text) == condition_rows[
            "hypothesis_generated"
        ].fillna("").map(normalize_text)
        comparisons.append(
            {
                "condition": condition,
                "n": int(len(condition_rows)),
                "actual_mean_WER": actual_mean,
                "baseline_mean_WER": baseline_mean,
                "expected_mean_WER": expected_mean,
                "abs_delta_vs_expected_WER": abs(actual_mean - expected_mean),
                "abs_delta_vs_baseline_WER": abs(actual_mean - baseline_mean),
                "raw_hypothesis_match_rate": float(exact_hyp_match.mean()),
                "normalized_hypothesis_match_rate": float(norm_hyp_match.mean()),
                "passes_tolerance": abs(actual_mean - expected_mean) <= tolerance,
            }
        )
    passes = not missing_keys and all(row.get("passes_tolerance", False) for row in comparisons)
    return {
        "repetition_penalty": 1.0,
        "baseline_rows": int(len(baseline_clean)),
        "generated_rows": int(len(generated_clean)),
        "matched_rows": int(len(matched)),
        "missing_or_extra_key_count": int(len(missing_keys)),
        "missing_or_extra_key_examples": missing_keys[:20],
        "comparisons": comparisons,
        "passes": bool(passes),
        "tolerance": tolerance,
    }


def add_split_column(frame: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(splits[["utterance_id", "split"]], on="utterance_id", how="left")
    if out["split"].isna().any():
        raise ValueError("Some rows were not assigned a split")
    return out


def make_before_after(before: pd.DataFrame, after: pd.DataFrame, *, split: str, penalty_before: float, penalty_after: float) -> pd.DataFrame:
    before_keys = before[KEY_COLUMNS].copy()
    after_keys = after[KEY_COLUMNS].copy()
    if before_keys.duplicated().any() or after_keys.duplicated().any():
        raise ValueError("Before/after rows must be unique by row key")
    before_renamed = before[KEY_COLUMNS + ["hypothesis"] + METRIC_COLUMNS].rename(
        columns={"hypothesis": "hypothesis_before", **{column: f"{column}_before" for column in METRIC_COLUMNS}}
    )
    after_renamed = after[KEY_COLUMNS + ["hypothesis"] + METRIC_COLUMNS].rename(
        columns={"hypothesis": "hypothesis_after", **{column: f"{column}_after" for column in METRIC_COLUMNS}}
    )
    merged = before_renamed.merge(after_renamed, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError(f"Before/after key mismatch: before={len(before)} after={len(after)} merged={len(merged)}")
    merged["split"] = split
    merged["repetition_penalty_before"] = penalty_before
    merged["repetition_penalty_after"] = penalty_after
    return merged


def add_perturbation_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = {item.label: item for item in DEFAULT_PERTURBATIONS}
    out = frame.copy()
    out["perturbation_type"] = out["perturbation"].map(lambda label: metadata[str(label)].perturb_type)
    out["amplitude"] = out["perturbation"].map(lambda label: metadata[str(label)].amplitude)
    out["duration"] = out["perturbation"].map(lambda label: metadata[str(label)].duration)
    return out


def summarize_before_after(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for group_values, group in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        row["N"] = int(len(group))
        row["WER_before"] = float(group["WER_before"].astype(float).mean())
        row["WER_after"] = float(group["WER_after"].astype(float).mean())
        row["delta_WER"] = row["WER_after"] - row["WER_before"]
        row["Hallucination_before"] = float(group["hallucination_like_before"].astype(bool).mean())
        row["Hallucination_after"] = float(group["hallucination_like_after"].astype(bool).mean())
        row["delta_Hallucination"] = row["Hallucination_after"] - row["Hallucination_before"]
        row["Grounding_before"] = float(group["grounding_gap_before"].astype(float).mean())
        row["Grounding_after"] = float(group["grounding_gap_after"].astype(float).mean())
        row["delta_Grounding"] = row["Grounding_after"] - row["Grounding_before"]
        row["Rep3_before"] = float(group["rep3_before"].astype(float).mean())
        row["Rep3_after"] = float(group["rep3_after"].astype(float).mean())
        row["delta_Rep3"] = row["Rep3_after"] - row["Rep3_before"]
        row["Rep4_before"] = float(group["rep4_before"].astype(float).mean())
        row["Rep4_after"] = float(group["rep4_after"].astype(float).mean())
        row["delta_Rep4"] = row["Rep4_after"] - row["Rep4_before"]
        row["relative_Rep4_reduction"] = np.nan if row["Rep4_before"] == 0 else (row["Rep4_before"] - row["Rep4_after"]) / row["Rep4_before"]
        rows.append(row)
    return pd.DataFrame(rows)


def select_global_penalty(grid: pd.DataFrame) -> Dict[str, object]:
    candidates = []
    for penalty, penalty_rows in grid.groupby("repetition_penalty"):
        base = penalty_rows[(penalty_rows["condition"] == "Base") & (penalty_rows["perturbation"] == "none")]
        rr = penalty_rows[(penalty_rows["condition"] == "RR") & (penalty_rows["perturbation"] == "none")]
        if base.empty or rr.empty:
            continue
        base_wer_degradation = float(base.iloc[0]["delta_WER"])
        rr_rep4_reduction = float(rr.iloc[0]["relative_Rep4_reduction"])
        candidates.append(
            {
                "repetition_penalty": float(penalty),
                "base_clean_WER_degradation": base_wer_degradation,
                "rr_relative_Rep4_reduction": rr_rep4_reduction,
            }
        )
    primary = [
        row for row in candidates
        if row["rr_relative_Rep4_reduction"] >= 0.30 and row["base_clean_WER_degradation"] <= 0.005
    ]
    if primary:
        selected = sorted(primary, key=lambda row: row["repetition_penalty"])[0]
        return {"selected": True, "selection_rule": "primary", **selected, "candidates": candidates}
    fallback = [row for row in candidates if row["base_clean_WER_degradation"] <= 0.010]
    if fallback:
        selected = sorted(fallback, key=lambda row: (-row["rr_relative_Rep4_reduction"], row["repetition_penalty"]))[0]
        return {"selected": True, "selection_rule": "fallback", **selected, "candidates": candidates}
    return {"selected": False, "selection_rule": "none", "candidates": candidates}


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10000, seed: int = 1729) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def mcnemar_exact(before: Sequence[object], after: Sequence[object]) -> Dict[str, float]:
    before_bool = np.asarray(before, dtype=bool)
    after_bool = np.asarray(after, dtype=bool)
    before_only = int(np.logical_and(before_bool, ~after_bool).sum())
    after_only = int(np.logical_and(~before_bool, after_bool).sum())
    discordant = before_only + after_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(before_only, after_only) + 1)) * (0.5 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"mcnemar_before_only": before_only, "mcnemar_after_only": after_only, "mcnemar_p_value": float(p_value)}


def wilcoxon_pvalue(before: Sequence[float], after: Sequence[float]) -> float:
    diff = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.allclose(diff, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(diff).pvalue)
    except Exception:
        return np.nan


def paired_statistics(frame: pd.DataFrame, group_cols: Sequence[str], n_resamples: int = 10000, seed: int = 1729) -> pd.DataFrame:
    rows = []
    for group_values, group in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(group_cols, group_values))
        base["N"] = int(len(group))
        for metric in DELTA_METRICS:
            before_col = f"{metric}_before"
            after_col = f"{metric}_after"
            if metric == "hallucination_like":
                deltas = group[after_col].astype(bool).astype(int).to_numpy() - group[before_col].astype(bool).astype(int).to_numpy()
            else:
                deltas = group[after_col].astype(float).to_numpy() - group[before_col].astype(float).to_numpy()
            ci = bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed)
            metric_row = {**base, "metric": metric, **ci}
            if metric == "hallucination_like":
                metric_row.update(mcnemar_exact(group[before_col], group[after_col]))
            if metric == "grounding_gap":
                metric_row["wilcoxon_p_value"] = wilcoxon_pvalue(group[before_col], group[after_col])
            rows.append(metric_row)
    return pd.DataFrame(rows)


def write_json(path: Path | str, payload: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_fingerprint(path: Path | str) -> Dict[str, object]:
    file_path = Path(path)
    stat = file_path.stat()
    return {"path": str(file_path), "size": stat.st_size, "mtime": stat.st_mtime}


def freeze_config(selection: Dict[str, object], output_json: Path | str, *, dev_grid_csv: Path | str, splits_csv: Path | str, baseline_csv: Path | str) -> Dict[str, object]:
    if not selection.get("selected"):
        raise ValueError("Cannot freeze repetition penalty because selection did not succeed")
    payload = {
        "selected_repetition_penalty": float(selection["repetition_penalty"]),
        "selection_rule": selection["selection_rule"],
        "base_clean_WER_degradation": float(selection["base_clean_WER_degradation"]),
        "rr_relative_Rep4_reduction": float(selection["rr_relative_Rep4_reduction"]),
        "dev_grid_csv": str(dev_grid_csv),
        "splits_csv": str(splits_csv),
        "baseline_csv": file_fingerprint(baseline_csv),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": selection.get("candidates", []),
    }
    write_json(output_json, payload)
    return payload


def run_verify(args: argparse.Namespace) -> None:
    baseline = read_clean_baseline(args.baseline_csv)
    if args.generated_csv.exists() and not args.force_decode:
        generated = pd.read_csv(args.generated_csv)
    else:
        utterance_ids = sorted(baseline["utterance_id"].astype(str).unique())
        if args.max_samples:
            utterance_ids = utterance_ids[: args.max_samples]
        generated = decode_and_score(
            utterance_ids=utterance_ids,
            conditions=args.conditions,
            perturbations=[Perturbation("none", "none", 0.0, 0.0)],
            repetition_penalty=1.0,
            output_csv=args.generated_csv,
            test_tsv=args.test_tsv,
            clips_dir=args.clips_dir,
            batch_size=args.batch_size,
            lm_batch_size=args.lm_batch_size,
            ctc_cache_path=args.ctc_cache_path,
            thresholds_json=args.thresholds_json,
            threshold_source_csv=args.threshold_source_csv,
            score_lms=not args.skip_lm_scoring,
            score_ctc=not args.skip_ctc_scoring,
        )
    report = verify_reproduction(generated, baseline, tolerance=args.tolerance)
    write_json(args.report_json, report)
    if not report["passes"]:
        raise SystemExit("repetition_penalty=1.00 reproduction failed; stopping before DEV grid")


def run_dev_grid(args: argparse.Namespace) -> None:
    baseline = read_clean_baseline(args.baseline_csv)
    splits = load_or_create_splits(baseline, args.splits_csv, args.dev_fraction, args.split_seed)
    baseline = add_split_column(baseline, splits)
    dev_before = baseline[baseline["split"] == "dev"].copy()
    utterance_ids = sorted(dev_before["utterance_id"].astype(str).unique())
    if args.max_samples:
        utterance_ids = utterance_ids[: args.max_samples]
    frames = []
    joined_frames = []
    for penalty in args.penalties:
        decoded = decode_and_score(
            utterance_ids=utterance_ids,
            conditions=args.conditions,
            perturbations=[Perturbation("none", "none", 0.0, 0.0)],
            repetition_penalty=penalty,
            test_tsv=args.test_tsv,
            clips_dir=args.clips_dir,
            batch_size=args.batch_size,
            lm_batch_size=args.lm_batch_size,
            ctc_cache_path=args.ctc_cache_path,
            thresholds_json=args.thresholds_json,
            threshold_source_csv=args.threshold_source_csv,
            score_lms=not args.skip_lm_scoring,
            score_ctc=not args.skip_ctc_scoring,
        )
        decoded["split"] = "dev"
        frames.append(decoded)
        before_subset = dev_before[dev_before["utterance_id"].astype(str).isin(utterance_ids)].copy()
        joined = make_before_after(before_subset, decoded, split="dev", penalty_before=1.0, penalty_after=penalty)
        joined["repetition_penalty"] = penalty
        joined_frames.append(joined)
    outputs = pd.concat(frames, ignore_index=True)
    outputs.to_csv(args.outputs_csv, index=False)
    joined_all = pd.concat(joined_frames, ignore_index=True)
    grid = summarize_before_after(joined_all, ["repetition_penalty", "condition", "perturbation"])
    grid.to_csv(args.grid_csv, index=False)


def run_select(args: argparse.Namespace) -> None:
    grid = pd.read_csv(args.grid_csv)
    selection = select_global_penalty(grid)
    write_json(args.selection_json, selection)
    if not selection.get("selected"):
        raise SystemExit("No repetition penalty satisfied the primary or fallback selection rule")
    freeze_config(selection, args.config_json, dev_grid_csv=args.grid_csv, splits_csv=args.splits_csv, baseline_csv=args.baseline_csv)


def load_config(path: Path | str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_clean_test(args: argparse.Namespace) -> None:
    config = load_config(args.config_json)
    penalty = float(config["selected_repetition_penalty"])
    baseline = read_clean_baseline(args.baseline_csv)
    splits = load_or_create_splits(baseline, args.splits_csv, args.dev_fraction, args.split_seed)
    baseline = add_split_column(baseline, splits)
    test_before = baseline[baseline["split"] == "test"].copy()
    utterance_ids = sorted(test_before["utterance_id"].astype(str).unique())
    if args.max_samples:
        utterance_ids = utterance_ids[: args.max_samples]
    decoded = decode_and_score(
        utterance_ids=utterance_ids,
        conditions=args.conditions,
        perturbations=[Perturbation("none", "none", 0.0, 0.0)],
        repetition_penalty=penalty,
        output_csv=args.outputs_csv,
        test_tsv=args.test_tsv,
        clips_dir=args.clips_dir,
        batch_size=args.batch_size,
        lm_batch_size=args.lm_batch_size,
        ctc_cache_path=args.ctc_cache_path,
        thresholds_json=args.thresholds_json,
        threshold_source_csv=args.threshold_source_csv,
        score_lms=not args.skip_lm_scoring,
        score_ctc=not args.skip_ctc_scoring,
    )
    before_subset = test_before[test_before["utterance_id"].astype(str).isin(utterance_ids)].copy()
    joined = make_before_after(before_subset, decoded, split="test", penalty_before=1.0, penalty_after=penalty)
    joined.to_csv(args.before_after_csv, index=False)
    summary = summarize_before_after(joined, ["condition"])
    summary.to_csv(args.summary_csv, index=False)
    stats = paired_statistics(joined, ["condition"], n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    stats.to_csv(args.stats_csv, index=False)


def run_perturbed_test(args: argparse.Namespace) -> None:
    config = load_config(args.config_json)
    penalty = float(config["selected_repetition_penalty"])
    baseline = read_clean_baseline(args.baseline_csv)
    splits = load_or_create_splits(baseline, args.splits_csv, args.dev_fraction, args.split_seed)
    baseline = add_split_column(baseline, splits)
    test_rows = baseline[baseline["split"] == "test"].copy()
    utterance_ids = sorted(test_rows["utterance_id"].astype(str).unique())
    if args.max_samples:
        utterance_ids = utterance_ids[: args.max_samples]
    perturbations = [item for item in selected_perturbations(args.perturbations) if item.label != "none"]
    before = decode_and_score(
        utterance_ids=utterance_ids,
        conditions=args.conditions,
        perturbations=perturbations,
        repetition_penalty=1.0,
        test_tsv=args.test_tsv,
        clips_dir=args.clips_dir,
        batch_size=args.batch_size,
        lm_batch_size=args.lm_batch_size,
        ctc_cache_path=args.ctc_cache_path,
        thresholds_json=args.thresholds_json,
        threshold_source_csv=args.threshold_source_csv,
        score_lms=not args.skip_lm_scoring,
        score_ctc=not args.skip_ctc_scoring,
        perturbation_seed=args.perturbation_seed,
    )
    after = decode_and_score(
        utterance_ids=utterance_ids,
        conditions=args.conditions,
        perturbations=perturbations,
        repetition_penalty=penalty,
        test_tsv=args.test_tsv,
        clips_dir=args.clips_dir,
        batch_size=args.batch_size,
        lm_batch_size=args.lm_batch_size,
        ctc_cache_path=args.ctc_cache_path,
        thresholds_json=args.thresholds_json,
        threshold_source_csv=args.threshold_source_csv,
        score_lms=not args.skip_lm_scoring,
        score_ctc=not args.skip_ctc_scoring,
        perturbation_seed=args.perturbation_seed,
    )
    joined = make_before_after(before, after, split="test_perturbed", penalty_before=1.0, penalty_after=penalty)
    joined = add_perturbation_metadata(joined)
    joined.to_csv(args.before_after_csv, index=False)
    summary = summarize_before_after(joined, ["condition", "perturbation", "perturbation_type", "amplitude", "duration"])
    summary.to_csv(args.summary_csv, index=False)
    stats = paired_statistics(
        joined,
        ["condition", "perturbation", "perturbation_type", "amplitude", "duration"],
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    stats.to_csv(args.stats_csv, index=False)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--thresholds_json", type=Path, default=DEFAULT_THRESHOLDS_JSON)
    parser.add_argument("--threshold_source_csv", type=Path, default=DEFAULT_THRESHOLD_SOURCE_CSV)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--ctc_cache_path", type=Path, default=DEFAULT_CTC_CACHE)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lm_batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_lm_scoring", action="store_true")
    parser.add_argument("--skip_ctc_scoring", action="store_true")


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--splits_csv", type=Path, default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=1729)


def add_stats_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=1729)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 1 repetition-penalty mitigation phases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-1.00")
    add_common_args(verify)
    verify.add_argument("--generated_csv", type=Path, default=DEFAULT_RESULTS_DIR / "repetition_penalty_1.00_outputs.csv")
    verify.add_argument("--report_json", type=Path, default=DEFAULT_REPRO_REPORT)
    verify.add_argument("--tolerance", type=float, default=0.02)
    verify.add_argument("--force_decode", action="store_true")
    verify.set_defaults(func=run_verify)

    dev = subparsers.add_parser("dev-grid")
    add_common_args(dev)
    add_split_args(dev)
    dev.add_argument("--penalties", nargs="+", type=float, default=CANDIDATE_PENALTIES)
    dev.add_argument("--outputs_csv", type=Path, default=DEFAULT_DEV_OUTPUTS)
    dev.add_argument("--grid_csv", type=Path, default=DEFAULT_DEV_GRID)
    dev.set_defaults(func=run_dev_grid)

    select = subparsers.add_parser("select")
    select.add_argument("--grid_csv", type=Path, default=DEFAULT_DEV_GRID)
    select.add_argument("--selection_json", type=Path, default=DEFAULT_RESULTS_DIR / "repetition_penalty_selection.json")
    select.add_argument("--config_json", type=Path, default=DEFAULT_CONFIG)
    select.add_argument("--splits_csv", type=Path, default=DEFAULT_SPLITS_CSV)
    select.add_argument("--baseline_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    select.set_defaults(func=run_select)

    clean_test = subparsers.add_parser("clean-test")
    add_common_args(clean_test)
    add_split_args(clean_test)
    add_stats_args(clean_test)
    clean_test.add_argument("--config_json", type=Path, default=DEFAULT_CONFIG)
    clean_test.add_argument("--outputs_csv", type=Path, default=DEFAULT_CLEAN_TEST_OUTPUTS)
    clean_test.add_argument("--before_after_csv", type=Path, default=DEFAULT_CLEAN_BEFORE_AFTER)
    clean_test.add_argument("--summary_csv", type=Path, default=DEFAULT_CLEAN_SUMMARY)
    clean_test.add_argument("--stats_csv", type=Path, default=DEFAULT_CLEAN_STATS)
    clean_test.set_defaults(func=run_clean_test)

    perturbed = subparsers.add_parser("perturbed-test")
    add_common_args(perturbed)
    add_split_args(perturbed)
    add_stats_args(perturbed)
    perturbed.add_argument("--config_json", type=Path, default=DEFAULT_CONFIG)
    perturbed.add_argument("--perturbations", nargs="+", default=[item.label for item in DEFAULT_PERTURBATIONS if item.label != "none"])
    perturbed.add_argument("--perturbation_seed", type=int, default=8675309)
    perturbed.add_argument("--before_after_csv", type=Path, default=DEFAULT_PERTURBED_BEFORE_AFTER)
    perturbed.add_argument("--summary_csv", type=Path, default=DEFAULT_PERTURBED_SUMMARY)
    perturbed.add_argument("--stats_csv", type=Path, default=DEFAULT_PERTURBED_STATS)
    perturbed.set_defaults(func=run_perturbed_test)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
