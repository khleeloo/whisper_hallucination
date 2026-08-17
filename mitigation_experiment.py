#!/usr/bin/env python3
"""Before/after mitigation experiment infrastructure.

This module intentionally does not implement a mitigation. It provides a shared
metric function for generated hypotheses and a baseline runner that decodes the
existing Whisper checkpoints without changing checkpoints, thresholds, or
existing result files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torchaudio

from acoustic_grounding_validation import DEFAULT_MODEL_NAME as DEFAULT_WAV2VEC2_MODEL
from acoustic_grounding_validation import Wav2Vec2CtcScorer
from evaluate_whisper_validation import (
    compute_lm_scores_cached,
    compute_repetition_metrics,
    compute_wer_metrics,
    load_test_data,
    normalize_text,
    validate_adapter_files,
)


SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_TEST_TSV = Path("/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv")
DEFAULT_CLIPS_DIR = Path("/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips")
DEFAULT_OUTPUT_CSV = Path("results/mitigation/baseline_outputs.csv")
DEFAULT_REPRO_REPORT = Path("results/mitigation/baseline_reproduction_report.json")
DEFAULT_CTC_CACHE = Path("results/mitigation/wav2vec2_ctc_score_cache.jsonl")
DEFAULT_THRESHOLDS_JSON = Path("results/mitigation/frozen_hallucination_thresholds.json")
DEFAULT_THRESHOLD_SOURCE_CSV = SCRATCH_ROOT / "eval_validation" / "per_utterance_base_ckpt14000.csv"
DEFAULT_BASE_MODEL = "openai/whisper-large-v3"
DEFAULT_GPT2_MODEL = "gpt2"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CRITERION_NAME = "mean_WER_mean_Qwen3_clean_base"

OUTPUT_COLUMNS = [
    "utterance_id",
    "condition",
    "perturbation",
    "reference",
    "hypothesis",
    "WER",
    "qwen_plaus",
    "gpt2_plaus",
    "rep2",
    "rep3",
    "rep4",
    "hallucination_like",
    "ctc_nll",
    "grounding_gap",
]

DEFAULT_MODEL_CONFIGS = {
    "Base": {"model_dir": SCRATCH_ROOT / "base" / "checkpoint-10000", "noise_ratio": 0.0},
    "RR": {"model_dir": SCRATCH_ROOT / "rr_64pct" / "checkpoint-9375", "noise_ratio": 0.64},
    "RU": {"model_dir": SCRATCH_ROOT / "ru_64pct" / "checkpoint-9375", "noise_ratio": 0.64},
    "UR": {"model_dir": SCRATCH_ROOT / "ur_64pct" / "checkpoint-10000", "noise_ratio": 0.64},
    "UU": {"model_dir": SCRATCH_ROOT / "uu_64pct" / "final", "noise_ratio": 0.64},
}


@dataclass(frozen=True)
class Perturbation:
    label: str
    perturb_type: str
    amplitude: float = 0.0
    duration: float = 0.0


DEFAULT_PERTURBATIONS = [
    Perturbation("none", "none", 0.0, 0.0),
    Perturbation("onset_noise_amp0.05_dur0.5", "onset_noise", 0.05, 0.5),
    Perturbation("onset_noise_amp0.5_dur0.5", "onset_noise", 0.5, 0.5),
    Perturbation("onset_noise_amp0.75_dur0.5", "onset_noise", 0.75, 0.5),
    Perturbation("full_noise_amp0.5_dur0.0", "full_noise", 0.5, 0.0),
    Perturbation("full_noise_amp0.75_dur0.0", "full_noise", 0.75, 0.0),
    Perturbation("reverb_amp0.5_dur0.5", "reverb", 0.5, 0.5),
    Perturbation("reverb_amp0.8_dur0.5", "reverb", 0.8, 0.5),
    Perturbation("leading_silence_amp0.0_dur1.0", "leading_silence", 0.0, 1.0),
    Perturbation("leading_silence_amp0.0_dur3.0", "leading_silence", 0.0, 3.0),
    Perturbation("silence_amp0.0_dur0.0", "silence", 0.0, 0.0),
    Perturbation("speech_band_noise_amp0.5_dur0.0", "speech_band_noise", 0.5, 0.0),
    Perturbation("speech_band_noise_amp0.75_dur0.0", "speech_band_noise", 0.75, 0.0),
]

ScoreLmFn = Callable[[Sequence[str], Sequence[str], str, str, int], Sequence[float]]
ScoreCtcFn = Callable[[pd.DataFrame, str, str, Path], pd.DataFrame]


def _ensure_same_length(**columns: Sequence[object]) -> None:
    lengths = {name: len(value) for name, value in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input columns must have the same length: {lengths}")


def _default_lm_plausibility(
    hypotheses: Sequence[str],
    references: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> Sequence[float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    norm_hyps = [normalize_text(text) for text in hypotheses]
    norm_refs = [normalize_text(text) for text in references]
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        hyp_scores, _ = compute_lm_scores_cached(norm_hyps, model, tokenizer, device=device, batch_size=batch_size)
        ref_scores, _ = compute_lm_scores_cached(norm_refs, model, tokenizer, device=device, batch_size=batch_size)
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return [min(1.0, max(0.0, hyp / (ref + 1e-8))) for hyp, ref in zip(hyp_scores, ref_scores)]


def _default_ctc_scores(df: pd.DataFrame, model_name: str, device: str, cache_path: Path) -> pd.DataFrame:
    scorer = Wav2Vec2CtcScorer(model_name, device=device)
    cache = _load_ctc_cache(cache_path)
    pending_cache_rows = []
    out = df.copy()
    hyp_ctc_nlls = []
    grounding_gaps = []
    for row in out.itertuples(index=False):
        audio = None
        scored = {}
        for role, text in [("ref", row.reference), ("hyp", row.hypothesis)]:
            key = _ctc_cache_key(row, role, str(text), model_name)
            cached = cache.get(key)
            if cached is None:
                if audio is None:
                    audio = _load_perturbed_audio(str(row.audio_path), parse_perturbation(str(row.perturbation)))
                nll, token_count, normalized_text = scorer.ctc_nll(audio, str(text))
                cached = {
                    "cache_key": key,
                    "audio_path": str(row.audio_path),
                    "utterance_id": str(row.utterance_id),
                    "perturbation": str(row.perturbation),
                    "role": role,
                    "text": str(text),
                    "model_name": model_name,
                    "ctc_nll": nll,
                    "token_count": token_count,
                    "normalized_text": normalized_text,
                }
                cache[key] = cached
                pending_cache_rows.append(cached)
            scored[role] = cached
        ref_nll = float(scored["ref"]["ctc_nll"])
        hyp_nll = float(scored["hyp"]["ctc_nll"])
        hyp_ctc_nlls.append(hyp_nll)
        grounding_gaps.append(hyp_nll - ref_nll)
    _append_ctc_cache(cache_path, pending_cache_rows)
    out["hyp_ctc_nll"] = hyp_ctc_nlls
    out["grounding_gap"] = grounding_gaps
    return out


def _ctc_cache_key(row: object, role: str, text: str, model_name: str) -> str:
    payload = "||".join([
        str(getattr(row, "audio_path", "")),
        str(getattr(row, "utterance_id", "")),
        str(getattr(row, "perturbation", "")),
        role,
        text,
        model_name,
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_ctc_cache(cache_path: Path) -> Dict[str, Dict[str, object]]:
    if not cache_path.exists():
        return {}
    cache = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cache[str(item["cache_key"])] = item
    return cache


def _append_ctc_cache(cache_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_perturbed_audio(audio_path: str, perturbation: "Perturbation") -> np.ndarray:
    from evaluate_dual_metric import apply_perturbation

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = apply_perturbation(
        waveform,
        16000,
        perturbation.perturb_type,
        perturbation.amplitude,
        perturbation.duration,
    )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).detach().cpu().float().numpy()


def _find_qwen_plausibility_column(df: pd.DataFrame) -> str:
    preferred = "normalized_sentence_score_Qwen3-0.6B"
    if preferred in df.columns:
        return preferred
    candidates = [col for col in df.columns if "qwen" in col.lower() and "normalized" in col.lower()]
    if candidates:
        return candidates[0]
    candidates = [col for col in df.columns if "qwen" in col.lower()]
    if candidates:
        return candidates[0]
    raise ValueError("Could not find a Qwen/Qwen3 normalized plausibility column in clean Base source CSV")


def compute_frozen_hallucination_thresholds(
    source_csv: Path | str = DEFAULT_THRESHOLD_SOURCE_CSV,
    output_json: Path | str = DEFAULT_THRESHOLDS_JSON,
) -> Dict[str, object]:
    """Freeze the paper hallucination thresholds from clean Base only."""
    source_path = Path(source_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing clean Base threshold source CSV: {source_path}")
    base = pd.read_csv(source_path)
    if "wer" in base.columns:
        wer_col = "wer"
    elif "WER" in base.columns:
        wer_col = "WER"
    else:
        raise ValueError(f"Clean Base source CSV has no WER column: {source_path}")
    plausibility_col = _find_qwen_plausibility_column(base)
    thresholds = {
        "criterion_name": DEFAULT_CRITERION_NAME,
        "source_split": "clean Base validation/test evaluation",
        "source_condition": "Base",
        "source_csv": str(source_path),
        "N": int(len(base)),
        "wer_source_column": wer_col,
        "plausibility_source_column": plausibility_col,
        "wer_threshold": float(base[wer_col].astype(float).mean()),
        "qwen_plausibility_threshold": float(base[plausibility_col].astype(float).mean()),
        "comparison": "WER > wer_threshold and qwen_plaus > qwen_plausibility_threshold",
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return thresholds


def load_or_create_frozen_hallucination_thresholds(
    thresholds_json: Path | str = DEFAULT_THRESHOLDS_JSON,
    source_csv: Path | str = DEFAULT_THRESHOLD_SOURCE_CSV,
) -> Dict[str, object]:
    thresholds_path = Path(thresholds_json)
    if thresholds_path.exists():
        return json.loads(thresholds_path.read_text(encoding="utf-8"))
    return compute_frozen_hallucination_thresholds(source_csv, thresholds_path)


def apply_frozen_hallucination_thresholds(
    df: pd.DataFrame,
    thresholds: Dict[str, object],
    plausibility_col: str = "qwen_plaus",
) -> pd.Series:
    """Apply the clean-Base frozen high-WER/high-Qwen criterion to any rows."""
    wer_threshold = float(thresholds["wer_threshold"])
    plausibility_threshold = float(thresholds["qwen_plausibility_threshold"])
    return (df["WER"].astype(float) > wer_threshold) & (
        df[plausibility_col].astype(float) > plausibility_threshold
    )


def evaluate_hypotheses(
    utterance_ids: Sequence[str],
    conditions: Sequence[str],
    perturbations: Sequence[str],
    references: Sequence[str],
    hypotheses: Sequence[str],
    audio_paths: Sequence[str],
    *,
    device: Optional[str] = None,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    gpt2_model: str = DEFAULT_GPT2_MODEL,
    lm_batch_size: int = 4,
    wav2vec2_model: str = DEFAULT_WAV2VEC2_MODEL,
    ctc_cache_path: Path | str = DEFAULT_CTC_CACHE,
    score_lms: bool = True,
    score_ctc: bool = True,
    lm_score_fn: Optional[ScoreLmFn] = None,
    ctc_score_fn: Optional[ScoreCtcFn] = None,
    hallucination_thresholds: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Score already-generated hypotheses with the existing evaluation metrics."""
    _ensure_same_length(
        utterance_ids=utterance_ids,
        conditions=conditions,
        perturbations=perturbations,
        references=references,
        hypotheses=hypotheses,
        audio_paths=audio_paths,
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wer_rows = compute_wer_metrics(hypotheses, references)
    rep_rows = [compute_repetition_metrics(hypothesis) for hypothesis in hypotheses]
    df = pd.DataFrame(
        {
            "utterance_id": list(utterance_ids),
            "condition": list(conditions),
            "perturbation": list(perturbations),
            "audio_path": list(audio_paths),
            "reference": list(references),
            "hypothesis": list(hypotheses),
            "WER": [row["wer"] for row in wer_rows],
            "rep2": [row["bigram_rep_count"] for row in rep_rows],
            "rep3": [row["trigram_rep_count"] for row in rep_rows],
            "rep4": [row["fourgram_rep_count"] for row in rep_rows],
        }
    )
    if score_lms:
        scorer = lm_score_fn or _default_lm_plausibility
        df["qwen_plaus"] = list(scorer(hypotheses, references, qwen_model, device, lm_batch_size))
        df["gpt2_plaus"] = list(scorer(hypotheses, references, gpt2_model, device, lm_batch_size))
    else:
        df["qwen_plaus"] = np.nan
        df["gpt2_plaus"] = np.nan
    if df["qwen_plaus"].notna().any():
        thresholds = hallucination_thresholds or load_or_create_frozen_hallucination_thresholds()
        df["hallucination_like"] = apply_frozen_hallucination_thresholds(df, thresholds)
    else:
        df["hallucination_like"] = False
    if score_ctc:
        ctc_input = df[["utterance_id", "audio_path", "perturbation", "reference", "hypothesis", "WER"]].copy()
        ctc_scorer = ctc_score_fn or _default_ctc_scores
        scored = ctc_scorer(ctc_input, wav2vec2_model, device, Path(ctc_cache_path))
        df["ctc_nll"] = scored["hyp_ctc_nll"].astype(float).to_numpy()
        df["grounding_gap"] = scored["grounding_gap"].astype(float).to_numpy()
    else:
        df["ctc_nll"] = np.nan
        df["grounding_gap"] = np.nan
    return df[OUTPUT_COLUMNS].copy()


def parse_perturbation(label: str) -> Perturbation:
    if label == "none":
        return Perturbation("none", "none", 0.0, 0.0)
    if "_amp" not in label or "_dur" not in label:
        raise ValueError(f"Invalid perturbation label: {label}")
    perturb_type, rest = label.split("_amp", 1)
    amp_text, dur_text = rest.split("_dur", 1)
    return Perturbation(label, perturb_type, float(amp_text), float(dur_text))


def selected_perturbations(labels: Optional[Sequence[str]]) -> List[Perturbation]:
    if not labels:
        return list(DEFAULT_PERTURBATIONS)
    known = {item.label: item for item in DEFAULT_PERTURBATIONS}
    return [known.get(label, parse_perturbation(label)) for label in labels]


def _load_whisper_model(model_dir: Path, base_model_name: str, device: str):
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    validate_adapter_files(str(model_dir))
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name, dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, str(model_dir)).to(device)
    model.eval()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    processor = WhisperProcessor.from_pretrained(base_model_name, language="en", task="transcribe")
    return model, base_model, processor


def run_baseline(
    *,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
    test_tsv: Path | str = DEFAULT_TEST_TSV,
    clips_dir: Path | str = DEFAULT_CLIPS_DIR,
    conditions: Optional[Sequence[str]] = None,
    perturbation_labels: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    lm_batch_size: int = 4,
    base_model_name: str = DEFAULT_BASE_MODEL,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    gpt2_model: str = DEFAULT_GPT2_MODEL,
    wav2vec2_model: str = DEFAULT_WAV2VEC2_MODEL,
    ctc_cache_path: Path | str = DEFAULT_CTC_CACHE,
    thresholds_json: Path | str = DEFAULT_THRESHOLDS_JSON,
    threshold_source_csv: Path | str = DEFAULT_THRESHOLD_SOURCE_CSV,
    score_lms: bool = True,
    score_ctc: bool = True,
    repetition_penalty: float = 1.0,
    device: Optional[str] = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    condition_names = list(conditions or DEFAULT_MODEL_CONFIGS.keys())
    perturbations = selected_perturbations(perturbation_labels)
    samples = load_test_data(str(test_tsv), str(clips_dir), max_samples=max_samples)
    audio_paths = [sample["audio_path"] for sample in samples]
    references = [sample["reference"] for sample in samples]
    utterance_ids = [sample.get("utt_id") or Path(sample["audio_path"]).stem for sample in samples]
    frames = []
    hallucination_thresholds = load_or_create_frozen_hallucination_thresholds(thresholds_json, threshold_source_csv)
    from evaluate_dual_metric import transcribe_batch as transcribe_perturbed_batch

    for condition in condition_names:
        if condition not in DEFAULT_MODEL_CONFIGS:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {sorted(DEFAULT_MODEL_CONFIGS)}")
        config = DEFAULT_MODEL_CONFIGS[condition]
        model, base_model, processor = _load_whisper_model(Path(config["model_dir"]), base_model_name, device)
        try:
            for perturbation in perturbations:
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
                frames.append(
                    evaluate_hypotheses(
                        utterance_ids,
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
                        hallucination_thresholds=hallucination_thresholds,
                    )
                )
        finally:
            model.cpu()
            del model, base_model, processor
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def aggregate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.copy()
    grouped["WAcc"] = 1.0 - grouped["WER"].astype(float)
    grouped["hallucination_like"] = grouped["hallucination_like"].astype(bool)
    return grouped.groupby(["condition", "perturbation"], dropna=False).agg(
        n_samples=("utterance_id", "count"),
        mean_WER=("WER", "mean"),
        mean_WAcc=("WAcc", "mean"),
        mean_qwen_plaus=("qwen_plaus", "mean"),
        mean_gpt2_plaus=("gpt2_plaus", "mean"),
        mean_rep2=("rep2", "mean"),
        mean_rep3=("rep3", "mean"),
        mean_rep4=("rep4", "mean"),
        hallucination_like_rate=("hallucination_like", "mean"),
        mean_ctc_nll=("ctc_nll", "mean"),
        mean_grounding_gap=("grounding_gap", "mean"),
    ).reset_index()


def compare_clean_baseline(
    baseline_csv: Path | str = DEFAULT_OUTPUT_CSV,
    report_path: Path | str = DEFAULT_REPRO_REPORT,
    tolerance: float = 0.02,
) -> Dict[str, object]:
    baseline = pd.read_csv(baseline_csv)
    clean = baseline[baseline["perturbation"] == "none"].copy()
    aggregates = aggregate_baseline(clean)
    reference_files = {
        "Base": [SCRATCH_ROOT / "eval_validation" / "per_utterance_base_ckpt14000.csv"],
        "RR": [SCRATCH_ROOT / "eval_64pct" / "per_utterance_rr_64pct_checkpoint-9375.csv"],
        "RU": [SCRATCH_ROOT / "eval_64pct" / "per_utterance_ru_64pct_checkpoint-9375.csv"],
        "UR": [
            SCRATCH_ROOT / "eval_64pct" / "per_utterance_ur_64pct_checkpoint-10000_shard00-of-02.csv",
            SCRATCH_ROOT / "eval_64pct" / "per_utterance_ur_64pct_checkpoint-10000_shard01-of-02.csv",
        ],
        "UU": [
            SCRATCH_ROOT / "eval_64pct" / "per_utterance_uu_64pct_final_shard00-of-02.csv",
            SCRATCH_ROOT / "eval_64pct" / "per_utterance_uu_64pct_final_shard01-of-02.csv",
        ],
    }
    comparisons = []
    missing = []
    for condition, paths in reference_files.items():
        if not all(path.exists() for path in paths):
            missing.extend(str(path) for path in paths if not path.exists())
            continue
        reference = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        actual_row = aggregates[aggregates["condition"] == condition]
        if actual_row.empty:
            missing.append(f"baseline condition {condition}")
            continue
        actual_wer = float(actual_row.iloc[0]["mean_WER"])
        expected_wer = float(reference["wer"].mean())
        comparisons.append(
            {
                "condition": condition,
                "actual_mean_WER": actual_wer,
                "expected_mean_WER": expected_wer,
                "abs_delta_WER": abs(actual_wer - expected_wer),
                "passes_tolerance": abs(actual_wer - expected_wer) <= tolerance,
            }
        )
    report = {
        "baseline_csv": str(baseline_csv),
        "tolerance": tolerance,
        "comparisons": comparisons,
        "missing_reference_inputs": missing,
        "reproduces": bool(comparisons) and all(row["passes_tolerance"] for row in comparisons) and not missing,
    }
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_rows_csv(rows: Iterable[Dict[str, object]], output_csv: Path | str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-mitigation baseline outputs for mitigation experiments.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_MODEL_CONFIGS.keys()))
    parser.add_argument("--perturbations", nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lm_batch_size", type=int, default=4)
    parser.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--gpt2_model", default=DEFAULT_GPT2_MODEL)
    parser.add_argument("--wav2vec2_model", default=DEFAULT_WAV2VEC2_MODEL)
    parser.add_argument("--ctc_cache_path", type=Path, default=DEFAULT_CTC_CACHE)
    parser.add_argument("--thresholds_json", type=Path, default=DEFAULT_THRESHOLDS_JSON)
    parser.add_argument("--threshold_source_csv", type=Path, default=DEFAULT_THRESHOLD_SOURCE_CSV)
    parser.add_argument("--write_thresholds_only", action="store_true")
    parser.add_argument("--skip_lm_scoring", action="store_true")
    parser.add_argument("--skip_ctc_scoring", action="store_true")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--compare_only", action="store_true")
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPRO_REPORT)
    parser.add_argument("--reproduction_tolerance", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.write_thresholds_only:
        compute_frozen_hallucination_thresholds(args.threshold_source_csv, args.thresholds_json)
        return
    if not args.compare_only:
        run_baseline(
            output_csv=args.output_csv,
            test_tsv=args.test_tsv,
            clips_dir=args.clips_dir,
            conditions=args.conditions,
            perturbation_labels=args.perturbations,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            lm_batch_size=args.lm_batch_size,
            qwen_model=args.qwen_model,
            gpt2_model=args.gpt2_model,
            wav2vec2_model=args.wav2vec2_model,
            ctc_cache_path=args.ctc_cache_path,
            thresholds_json=args.thresholds_json,
            threshold_source_csv=args.threshold_source_csv,
            score_lms=not args.skip_lm_scoring,
            score_ctc=not args.skip_ctc_scoring,
            repetition_penalty=args.repetition_penalty,
        )
    compare_clean_baseline(args.output_csv, args.report_path, args.reproduction_tolerance)


if __name__ == "__main__":
    main()