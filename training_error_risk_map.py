#!/usr/bin/env python3
"""Evaluate how training-error structure changes Whisper failure phenotypes.

This is a checkpoint-only experiment: no training is performed.

Primary controlled analyses:
1. Structure at 64% corruption: RR vs RU vs UR vs UU.
2. Dose response: RR and RU at 16%, 32%, and 64%.

The clean Common Voice adapter is included as descriptive context only because
it was trained with a different LoRA/optimization recipe than the noisy sweep.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from evaluate_dual_metric import transcribe_batch as transcribe_perturbed_batch
from evaluate_whisper_validation import (
    compute_repetition_metrics,
    compute_wer_metrics,
    load_test_data,
    normalize_text,
)
from mitigation_experiment import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CLIPS_DIR,
    DEFAULT_QWEN_MODEL,
    DEFAULT_TEST_TSV,
    DEFAULT_THRESHOLD_SOURCE_CSV,
    DEFAULT_THRESHOLDS_JSON,
    _default_lm_plausibility,
    _load_whisper_model,
    apply_frozen_hallucination_thresholds,
    load_or_create_frozen_hallucination_thresholds,
    selected_perturbations,
)

SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DATA_ROOT = Path("/scratch/vemotionsys/rmfrieske/datasets")
DEFAULT_OUTPUT_DIR = SCRATCH_ROOT / "training_error_risk_map"
DEFAULT_PERTURBATIONS = ["none", "full_noise_amp0.5_dur0.0"]

MODEL_CONFIGS = {
    "clean_context": {
        "condition": "CLEAN",
        "ratio": 0.0,
        "adapter": SCRATCH_ROOT / "base" / "checkpoint-10000",
        "noisy_only": None,
        "analysis_role": "context_only",
    },
    "rr_16": {
        "condition": "RR",
        "ratio": 0.16,
        "adapter": SCRATCH_ROOT / "rr_16pct" / "checkpoint-1000",
        "noisy_only": DATA_ROOT / "whisper_hallucination_16pct" / "rr" / "noisy_only.tsv",
        "analysis_role": "dose_response",
    },
    "ru_16": {
        "condition": "RU",
        "ratio": 0.16,
        "adapter": SCRATCH_ROOT / "ru_16pct" / "checkpoint-1000",
        "noisy_only": DATA_ROOT / "whisper_hallucination_16pct" / "ru" / "noisy_only.tsv",
        "analysis_role": "dose_response",
    },
    "rr_32": {
        "condition": "RR",
        "ratio": 0.32,
        "adapter": SCRATCH_ROOT / "rr_32pct" / "checkpoint-2000",
        "noisy_only": DATA_ROOT / "whisper_hallucination_32pct" / "rr" / "noisy_only.tsv",
        "analysis_role": "dose_response",
    },
    "ru_32": {
        "condition": "RU",
        "ratio": 0.32,
        "adapter": SCRATCH_ROOT / "ru_32pct" / "checkpoint-2000",
        "noisy_only": DATA_ROOT / "whisper_hallucination_32pct" / "ru" / "noisy_only.tsv",
        "analysis_role": "dose_response",
    },
    "rr_64": {
        "condition": "RR",
        "ratio": 0.64,
        "adapter": SCRATCH_ROOT / "rr_64pct" / "checkpoint-9375",
        "noisy_only": DATA_ROOT / "whisper_hallucination_64pct" / "rr" / "noisy_only.tsv",
        "analysis_role": "structure_and_dose",
    },
    "ru_64": {
        "condition": "RU",
        "ratio": 0.64,
        "adapter": SCRATCH_ROOT / "ru_64pct" / "checkpoint-9375",
        "noisy_only": DATA_ROOT / "whisper_hallucination_64pct" / "ru" / "noisy_only.tsv",
        "analysis_role": "structure_and_dose",
    },
    "ur_64": {
        "condition": "UR",
        "ratio": 0.64,
        "adapter": SCRATCH_ROOT / "ur_64pct" / "checkpoint-10000",
        "noisy_only": DATA_ROOT / "whisper_hallucination_64pct" / "ur" / "noisy_only.tsv",
        "analysis_role": "structure",
    },
    "uu_64": {
        "condition": "UU",
        "ratio": 0.64,
        "adapter": SCRATCH_ROOT / "uu_64pct" / "final",
        "noisy_only": DATA_ROOT / "whisper_hallucination_64pct" / "uu" / "noisy_only.tsv",
        "analysis_role": "structure",
    },
}


def read_adapter_signature(adapter_dir: Path) -> Dict[str, object]:
    path = adapter_dir / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("target_modules", [])
    if isinstance(targets, list):
        targets = sorted(str(x) for x in targets)
    return {
        "r": payload.get("r"),
        "lora_alpha": payload.get("lora_alpha"),
        "target_modules": targets,
        "peft_type": payload.get("peft_type"),
    }


def preflight(selected: Dict[str, dict]) -> Dict[str, object]:
    rows = {}
    for name, cfg in selected.items():
        adapter = Path(cfg["adapter"])
        if not adapter.exists():
            raise FileNotFoundError(f"{name}: missing adapter directory: {adapter}")
        has_weights = (adapter / "adapter_model.safetensors").exists() or (adapter / "adapter_model.bin").exists()
        if not has_weights:
            raise FileNotFoundError(f"{name}: missing adapter weights in {adapter}")
        if cfg["noisy_only"] is not None and not Path(cfg["noisy_only"]).exists():
            raise FileNotFoundError(f"{name}: missing noisy_only.tsv: {cfg['noisy_only']}")
        rows[name] = {
            "adapter": str(adapter),
            "condition": cfg["condition"],
            "ratio": cfg["ratio"],
            "analysis_role": cfg["analysis_role"],
            "adapter_signature": read_adapter_signature(adapter),
        }

    structure_names = [n for n in ["rr_64", "ru_64", "ur_64", "uu_64"] if n in rows]
    if len(structure_names) > 1:
        signatures = [rows[n]["adapter_signature"] for n in structure_names]
        first = signatures[0]
        mismatched = [n for n, sig in zip(structure_names, signatures) if sig != first]
        if mismatched:
            raise RuntimeError(
                "64% structure adapters have mismatched LoRA configurations; "
                f"cannot treat as controlled comparison. Mismatched: {mismatched}. "
                f"Signatures: {json.dumps({n: rows[n]['adapter_signature'] for n in structure_names}, indent=2)}"
            )
    return rows


def score_asr_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    wer_rows = compute_wer_metrics(frame["hypothesis"].tolist(), frame["reference"].tolist())
    rep_rows = [compute_repetition_metrics(text) for text in frame["hypothesis"]]
    out = frame.copy()
    out["WER"] = [row["wer"] for row in wer_rows]
    out["rep3"] = [row["trigram_rep_count"] for row in rep_rows]
    out["rep4"] = [row["fourgram_rep_count"] for row in rep_rows]
    out["rep34"] = ((out["rep3"] > 0) | (out["rep4"] > 0)).astype(int)
    out["prediction_norm"] = [normalize_text(text) for text in out["hypothesis"]]
    return out


def load_corrupted_targets(path: Path):
    df = pd.read_csv(path, sep="\t")
    if "sentence" not in df.columns:
        raise ValueError(f"Missing sentence column in {path}")
    normalized = [normalize_text(str(x)) for x in df["sentence"].fillna("")]
    normalized = [x for x in normalized if x]
    counts = Counter(normalized)
    unique = list(counts.keys())
    if not unique:
        raise ValueError(f"No non-empty corrupted targets in {path}")
    return unique, counts


def add_target_provenance(
    frame: pd.DataFrame,
    noisy_only: Optional[Path],
    *,
    near_threshold: float,
    template_threshold: float,
    ngram_max: int,
    max_features: int,
) -> pd.DataFrame:
    out = frame.copy()
    if noisy_only is None:
        for col in [
            "best_corrupt_target_similarity",
            "best_corrupt_target_frequency",
            "exact_corrupt_target_reuse",
            "near_corrupt_target_reuse",
            "template_corrupt_target_reuse",
        ]:
            out[col] = np.nan
        out["best_corrupt_target"] = ""
        return out

    targets, counts = load_corrupted_targets(Path(noisy_only))
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, ngram_max),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    target_matrix = vectorizer.fit_transform(targets)
    query_matrix = vectorizer.transform(out["prediction_norm"].fillna("").tolist())
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(target_matrix)
    distances, indices = nn.kneighbors(query_matrix, return_distance=True)
    similarities = np.maximum(0.0, 1.0 - distances[:, 0])
    best_targets = [targets[int(idx)] for idx in indices[:, 0]]
    exact = [bool(pred and pred in counts) for pred in out["prediction_norm"].fillna("").tolist()]

    out["best_corrupt_target_similarity"] = similarities
    out["best_corrupt_target"] = best_targets
    out["best_corrupt_target_frequency"] = [counts[t] for t in best_targets]
    out["exact_corrupt_target_reuse"] = np.asarray(exact, dtype=int)
    out["near_corrupt_target_reuse"] = (similarities >= near_threshold).astype(int)
    out["template_corrupt_target_reuse"] = (similarities >= template_threshold).astype(int)
    return out


def safe_rate(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return float(valid.mean()) if len(valid) else float("nan")


def aggregate(frame: pd.DataFrame, wer_threshold: float) -> pd.DataFrame:
    rows = []
    for (model_name, condition, ratio, perturbation), g in frame.groupby(
        ["model_name", "condition", "corruption_ratio", "perturbation"], sort=False
    ):
        high_error = g["WER"] > wer_threshold
        hall = g["hallucination_like"].astype(bool)
        nonrep_hall = hall & (g["rep34"] == 0)
        pred_counts = g["prediction_norm"].value_counts(normalize=True)
        row = {
            "model_name": model_name,
            "condition": condition,
            "corruption_ratio": float(ratio),
            "perturbation": perturbation,
            "N": int(len(g)),
            "WER": float(g["WER"].mean()),
            "qwen_plaus": float(g["qwen_plaus"].mean()),
            "hallucination_like_rate": float(hall.mean()),
            "nonrep_hallucination_like_rate": float(nonrep_hall.mean()),
            "rep34_rate": float(g["rep34"].mean()),
            "top1_output_mass": float(pred_counts.iloc[0]) if len(pred_counts) else 0.0,
            "high_error_rate": float(high_error.mean()),
            "high_error_n": int(high_error.sum()),
        }
        if g["exact_corrupt_target_reuse"].notna().any():
            row.update({
                "exact_target_reuse_rate_all": safe_rate(g["exact_corrupt_target_reuse"]),
                "near_target_reuse_rate_all": safe_rate(g["near_corrupt_target_reuse"]),
                "template_target_reuse_rate_all": safe_rate(g["template_corrupt_target_reuse"]),
                "mean_best_target_similarity_all": safe_rate(g["best_corrupt_target_similarity"]),
                "exact_target_reuse_rate_high_error": safe_rate(g.loc[high_error, "exact_corrupt_target_reuse"]),
                "near_target_reuse_rate_high_error": safe_rate(g.loc[high_error, "near_corrupt_target_reuse"]),
                "template_target_reuse_rate_high_error": safe_rate(g.loc[high_error, "template_corrupt_target_reuse"]),
                "mean_best_target_similarity_high_error": safe_rate(g.loc[high_error, "best_corrupt_target_similarity"]),
                "exact_target_reuse_rate_hall_like": safe_rate(g.loc[hall, "exact_corrupt_target_reuse"]),
                "near_target_reuse_rate_hall_like": safe_rate(g.loc[hall, "near_corrupt_target_reuse"]),
                "mean_best_target_similarity_hall_like": safe_rate(g.loc[hall, "best_corrupt_target_similarity"]),
            })
        else:
            for col in [
                "exact_target_reuse_rate_all",
                "near_target_reuse_rate_all",
                "template_target_reuse_rate_all",
                "mean_best_target_similarity_all",
                "exact_target_reuse_rate_high_error",
                "near_target_reuse_rate_high_error",
                "template_target_reuse_rate_high_error",
                "mean_best_target_similarity_high_error",
                "exact_target_reuse_rate_hall_like",
                "near_target_reuse_rate_hall_like",
                "mean_best_target_similarity_hall_like",
            ]:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_conditions(requested: Optional[Iterable[str]]) -> Dict[str, dict]:
    if not requested:
        return dict(MODEL_CONFIGS)
    unknown = sorted(set(requested) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Choices: {sorted(MODEL_CONFIGS)}")
    return {name: MODEL_CONFIGS[name] for name in requested}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--qwen_batch_size", type=int, default=8)
    parser.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--perturbations", nargs="+", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--thresholds_json", type=Path, default=DEFAULT_THRESHOLDS_JSON)
    parser.add_argument("--threshold_source_csv", type=Path, default=DEFAULT_THRESHOLD_SOURCE_CSV)
    parser.add_argument("--near_threshold", type=float, default=0.85)
    parser.add_argument("--template_threshold", type=float, default=0.98)
    parser.add_argument("--tfidf_ngram_max", type=int, default=3)
    parser.add_argument("--tfidf_max_features", type=int, default=250000)
    args = parser.parse_args()

    selected = parse_conditions(args.conditions)
    preflight_rows = preflight(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_or_create_frozen_hallucination_thresholds(
        args.thresholds_json, args.threshold_source_csv
    )
    wer_threshold = float(thresholds["wer_threshold"])
    perturbations = selected_perturbations(args.perturbations)

    samples = load_test_data(str(args.test_tsv), str(args.clips_dir), max_samples=args.max_samples)
    if not samples:
        raise RuntimeError("No evaluation samples loaded.")
    audio_paths = [s["audio_path"] for s in samples]
    references = [s["reference"] for s in samples]
    utterance_ids = [s["utt_id"] for s in samples]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_frames: List[pd.DataFrame] = []
    for name, cfg in selected.items():
        print(
            f"\n=== {name}: {cfg['condition']} ratio={cfg['ratio']:.2f} adapter={cfg['adapter']} ===",
            flush=True,
        )
        model, base_model, processor = _load_whisper_model(
            Path(cfg["adapter"]), args.base_model, device
        )
        try:
            model_frames = []
            for perturb_idx, perturbation in enumerate(perturbations):
                matched_seed = args.seed + perturb_idx
                np.random.seed(matched_seed)
                torch.manual_seed(matched_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(matched_seed)
                hypotheses = transcribe_perturbed_batch(
                    model,
                    processor,
                    audio_paths,
                    perturb_type=perturbation.perturb_type,
                    perturb_amplitude=perturbation.amplitude,
                    perturb_duration=perturbation.duration,
                    device=device,
                    batch_size=args.batch_size,
                    repetition_penalty=1.0,
                )
                frame = pd.DataFrame({
                    "utterance_id": utterance_ids,
                    "model_name": name,
                    "condition": cfg["condition"],
                    "corruption_ratio": float(cfg["ratio"]),
                    "analysis_role": cfg["analysis_role"],
                    "perturbation": perturbation.label,
                    "reference": references,
                    "hypothesis": hypotheses,
                    "audio_path": audio_paths,
                })
                model_frames.append(score_asr_outputs(frame))
        finally:
            model.cpu()
            del model, base_model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_df = pd.concat(model_frames, ignore_index=True)
        print(f"Scoring Qwen plausibility for {len(model_df)} outputs...", flush=True)
        model_df["qwen_plaus"] = list(
            _default_lm_plausibility(
                model_df["hypothesis"].tolist(),
                model_df["reference"].tolist(),
                args.qwen_model,
                device,
                args.qwen_batch_size,
            )
        )
        model_df["hallucination_like"] = apply_frozen_hallucination_thresholds(
            model_df, thresholds, plausibility_col="qwen_plaus"
        ).astype(int)
        print("Computing corrupted-target provenance...", flush=True)
        model_df = add_target_provenance(
            model_df,
            Path(cfg["noisy_only"]) if cfg["noisy_only"] is not None else None,
            near_threshold=args.near_threshold,
            template_threshold=args.template_threshold,
            ngram_max=args.tfidf_ngram_max,
            max_features=args.tfidf_max_features,
        )
        all_frames.append(model_df)

    result = pd.concat(all_frames, ignore_index=True)
    per_path = args.output_dir / "per_utterance_training_error_risk_map.csv"
    result.to_csv(per_path, index=False)

    summary = aggregate(result, wer_threshold=wer_threshold)
    summary_path = args.output_dir / "summary_training_error_risk_map.csv"
    summary.to_csv(summary_path, index=False)

    structure = summary[
        (summary["corruption_ratio"] == 0.64)
        & summary["condition"].isin(["RR", "RU", "UR", "UU"])
    ].copy()
    structure_path = args.output_dir / "structure_64pct.csv"
    structure.to_csv(structure_path, index=False)

    dose = summary[
        summary["condition"].isin(["RR", "RU"])
        & summary["corruption_ratio"].isin([0.16, 0.32, 0.64])
    ].copy()
    dose_path = args.output_dir / "dose_response_rr_ru.csv"
    dose.to_csv(dose_path, index=False)

    manifest = {
        "experiment": "training_error_risk_map",
        "checkpoint_only": True,
        "base_model": args.base_model,
        "test_tsv": str(args.test_tsv),
        "clips_dir": str(args.clips_dir),
        "N_per_model_per_perturbation": len(samples),
        "seed": args.seed,
        "perturbations": [p.label for p in perturbations],
        "hallucination_thresholds": thresholds,
        "target_provenance": {
            "reference_set": "corrupted targets from noisy_only.tsv only",
            "near_tfidf_threshold": args.near_threshold,
            "template_tfidf_threshold": args.template_threshold,
            "tfidf_ngram_range": [1, args.tfidf_ngram_max],
        },
        "analysis_guards": {
            "clean_context_is_descriptive_only": True,
            "reason": (
                "clean Common Voice adapter used a different LoRA/optimization recipe "
                "than the noisy RR/RU/UR/UU sweep"
            ),
            "primary_structure_comparison": "RR vs RU vs UR vs UU at 64% corruption",
            "primary_dose_response": "RR and RU at 16%, 32%, 64%",
        },
        "models": preflight_rows,
        "outputs": {
            "per_utterance": str(per_path),
            "summary": str(summary_path),
            "structure_64pct": str(structure_path),
            "dose_response_rr_ru": str(dose_path),
        },
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    display_cols = [
        "model_name", "condition", "corruption_ratio", "perturbation", "N",
        "WER", "qwen_plaus", "hallucination_like_rate",
        "nonrep_hallucination_like_rate", "rep34_rate", "top1_output_mass",
        "exact_target_reuse_rate_high_error", "near_target_reuse_rate_high_error",
        "mean_best_target_similarity_high_error",
    ]
    print("\n=== Risk-map summary ===", flush=True)
    print(summary[display_cols].to_string(index=False), flush=True)
    print(f"\nSaved: {per_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {structure_path}", flush=True)
    print(f"Saved: {dose_path}", flush=True)
    print(f"Saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
