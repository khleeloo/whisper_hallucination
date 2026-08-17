"""
Validate hallucination-like ASR outputs with an independent wav2vec2 CTC model.

This script reuses saved hypotheses and per-utterance metrics, attaches audio paths
from a Common Voice-style manifest, computes normalized CTC NLL for references and
hypotheses, caches those scores, and runs WER-matched hallucination/control analyses.

Example:
    python acoustic_grounding_validation.py \
        --input_csv /scratch/vemotionsys/rmfrieske/whisper_hallucination/fairseq_eval_lm/per_utterance_metrics_fairseq.csv \
        --test_tsv /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv \
        --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/acoustic_validation \
        --sanity_only --max_score_rows 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio


DEFAULT_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_INPUT_CANDIDATES = [
    DEFAULT_ROOT / "eval_validation" / "per_utterance_metrics_whisper.csv",
    DEFAULT_ROOT / "fairseq_eval_lm" / "per_utterance_metrics_fairseq.csv",
    DEFAULT_ROOT / "fairseq_eval" / "per_utterance_metrics_fairseq.csv",
    Path("fairseq_eval_lm/per_utterance_metrics_fairseq.csv"),
    Path("fairseq_eval/per_utterance_metrics_fairseq.csv"),
]
DEFAULT_TEST_TSV = Path("/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv")
DEFAULT_CLIPS_DIR = Path("/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips")
DEFAULT_MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
CONDITION_ORDER = ["base", "rr", "ru", "ur", "uu"]
RNG_SEED = 1729


def normalize_condition(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text.lower() if text else "unknown"


def first_existing_path(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No input CSV found. Checked:\n  {joined}")


def select_input_csv(candidates: Sequence[Path], qwen_col: Optional[str], gpt2_col: Optional[str]) -> Path:
    rejected = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception as exc:
            rejected.append(f"{path}: could not read header ({exc})")
            continue
        has_qwen = find_score_column(header, qwen_col, ["normalized_sentence_score_qwen", "sentence_score_qwen", "qwen3", "qwen"])
        has_gpt2 = find_score_column(header, gpt2_col, ["normalized_sentence_score_gpt2", "sentence_score_gpt2", "gpt2"])
        has_audio_path = "audio_path" in header.columns
        if has_qwen and has_gpt2 and has_audio_path:
            return path
        missing = []
        if not has_qwen:
            missing.append("Qwen/Qwen3")
        if not has_gpt2:
            missing.append("GPT2")
        if not has_audio_path:
            missing.append("audio_path")
        rejected.append(f"{path}: missing {' and '.join(missing)} column")

    if rejected:
        joined = "\n  ".join(rejected)
        raise ValueError(
            "No auto-usable input CSV found with Qwen/Qwen3, GPT2, and audio_path columns. "
            "Pass --input_csv explicitly only when --test_tsv/--clips_dir belong to the same utterance order. Rejected:\n  "
            f"{joined}"
        )
    return first_existing_path(candidates)


def find_score_column(df: pd.DataFrame, preferred: Optional[str], patterns: Sequence[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    lowered = {col.lower(): col for col in df.columns}
    for pattern in patterns:
        for lower_col, original_col in lowered.items():
            if pattern.lower() in lower_col:
                return original_col
    return None


def find_score_columns(df: pd.DataFrame, preferred: Optional[str], patterns: Sequence[str]) -> List[str]:
    if preferred:
        return [preferred] if preferred in df.columns else []
    matches = []
    for col in df.columns:
        lower_col = col.lower()
        if any(pattern.lower() in lower_col for pattern in patterns):
            matches.append(col)
    return matches


def coalesce_numeric_columns(df: pd.DataFrame, columns: Sequence[str], output_col: str) -> pd.Series:
    if not columns:
        raise ValueError(f"Could not find source columns for {output_col}.")
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        values = values.fillna(pd.to_numeric(df[col], errors="coerce"))
    return values


def load_manifest_audio_paths(test_tsv: Path, clips_dir: Path) -> pd.DataFrame:
    rows = []
    with open(test_tsv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"Expected a Common Voice-style 'path' column in {test_tsv}")
        for idx, row in enumerate(reader):
            rel_path = row["path"]
            audio_path = clips_dir / rel_path
            utterance_id = row.get("client_id") or Path(rel_path).stem or f"utt_{idx:06d}"
            rows.append({
                "manifest_index": idx,
                "manifest_audio_path": str(audio_path),
                "manifest_utterance_id": str(utterance_id),
                "manifest_reference": row.get("sentence", ""),
            })
    if not rows:
        raise ValueError(f"No rows found in {test_tsv}")
    return pd.DataFrame(rows)


def comparable_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", str(value).lower())).strip()


def validate_manifest_alignment(df: pd.DataFrame, max_mismatch_rate: float) -> Dict[str, object]:
    if "manifest_reference" not in df.columns or "reference" not in df.columns:
        return {"checked": 0, "mismatches": 0, "mismatch_rate": 0.0}

    comparable = df[["reference", "manifest_reference"]].dropna().copy()
    if comparable.empty:
        return {"checked": 0, "mismatches": 0, "mismatch_rate": 0.0}

    reference_norm = comparable["reference"].map(comparable_text)
    manifest_norm = comparable["manifest_reference"].map(comparable_text)
    checked = int((reference_norm.str.len() > 0).sum())
    mismatches = int(((reference_norm != manifest_norm) & (reference_norm.str.len() > 0)).sum())
    mismatch_rate = float(mismatches / checked) if checked else 0.0
    if checked and mismatch_rate > max_mismatch_rate:
        raise ValueError(
            "Manifest/reference alignment check failed: "
            f"{mismatches}/{checked} rows mismatch ({mismatch_rate:.1%}), above --max_manifest_mismatch_rate={max_mismatch_rate:.1%}. "
            "This usually means reconstructed audio paths are not row-aligned with the input CSV."
        )
    if mismatches:
        print(f"Manifest/reference alignment warning: {mismatches}/{checked} rows mismatch ({mismatch_rate:.1%}).", flush=True)
    return {"checked": checked, "mismatches": mismatches, "mismatch_rate": mismatch_rate}


def attach_audio_paths(df: pd.DataFrame, manifest_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    if "audio_path" in out.columns and out["audio_path"].astype(str).str.len().gt(0).all():
        return out
    if manifest_df is None:
        raise ValueError("Input CSV has no audio_path column; provide --test_tsv and --clips_dir to reconstruct paths.")

    parts = []
    group_cols = [col for col in ["source_file", "model_name", "noise_condition"] if col in out.columns]
    if group_cols:
        grouped = out.groupby(group_cols, sort=False, dropna=False)
    else:
        grouped = [("all", out)]

    for _, group in grouped:
        if len(group) > len(manifest_df):
            raise ValueError(
                f"Cannot attach audio paths by row order: group has {len(group)} rows but manifest has {len(manifest_df)}."
            )
        merged = group.reset_index(drop=False).join(manifest_df.iloc[:len(group)].reset_index(drop=True))
        parts.append(merged.set_index("index"))

    out = pd.concat(parts).sort_index()
    out["audio_path"] = out["manifest_audio_path"]
    if "utterance_id" not in out.columns:
        out["utterance_id"] = out.get("utt_id", out["manifest_utterance_id"])
    return out


def standardize_columns(df: pd.DataFrame, qwen_col: Optional[str], gpt2_col: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    out = df.copy()
    rename_map = {}
    if "utt_id" in out.columns and "utterance_id" not in out.columns:
        rename_map["utt_id"] = "utterance_id"
    if "wer" in out.columns and "WER" not in out.columns:
        rename_map["wer"] = "WER"
    out = out.rename(columns=rename_map)

    required = ["utterance_id", "audio_path", "reference", "hypothesis", "WER"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns after reconstruction: {missing}")

    if "condition" not in out.columns:
        if "noise_condition" in out.columns:
            out["condition"] = out["noise_condition"].map(normalize_condition)
        elif "model_name" in out.columns:
            out["condition"] = out["model_name"].map(normalize_condition)
        else:
            out["condition"] = "unknown"
    out["condition"] = out["condition"].map(normalize_condition)

    if "perturbation_condition" not in out.columns:
        for candidate in ["perturb_type", "noise_condition", "condition"]:
            if candidate in out.columns:
                out["perturbation_condition"] = out[candidate].map(normalize_condition)
                break
        else:
            out["perturbation_condition"] = "unknown"

    qwen_score_cols = find_score_columns(
        out,
        qwen_col,
        ["normalized_sentence_score_qwen", "sentence_score_qwen", "qwen3", "qwen"],
    )
    gpt2_score_cols = find_score_columns(
        out,
        gpt2_col,
        ["normalized_sentence_score_gpt2", "sentence_score_gpt2", "gpt2"],
    )
    if not qwen_score_cols:
        raise ValueError("Could not find a Qwen/Qwen3 score column. Pass --qwen_col explicitly.")
    if not gpt2_score_cols:
        raise ValueError("Could not find a GPT2 score column. Pass --gpt2_col explicitly.")

    out["Qwen3_score"] = coalesce_numeric_columns(out, qwen_score_cols, "Qwen3_score")
    out["GPT2_score"] = coalesce_numeric_columns(out, gpt2_score_cols, "GPT2_score")
    out["WER"] = pd.to_numeric(out["WER"], errors="coerce")
    out["hypothesis"] = out["hypothesis"].fillna("").astype(str)
    out["reference"] = out["reference"].fillna("").astype(str)
    return out, "|".join(qwen_score_cols), "|".join(gpt2_score_cols)


def compute_audio_gate_features(waveform: torch.Tensor, sample_rate: int = 16000) -> Dict[str, float]:
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    samples = waveform.detach().float().reshape(-1).cpu()
    if samples.numel() == 0:
        return {"speech_fraction": 0.0, "snr_proxy_db": 0.0, "is_full_silence": True}

    frame_length = max(1, int(0.03 * sample_rate))
    hop_length = max(1, int(0.01 * sample_rate))
    if samples.numel() < frame_length:
        frames = samples.unsqueeze(0)
    else:
        frames = samples.unfold(0, frame_length, hop_length)
    frame_rms = torch.sqrt(frames.pow(2).mean(dim=1) + 1e-12).numpy()
    if frame_rms.size == 0:
        return {"speech_fraction": 0.0, "snr_proxy_db": 0.0, "is_full_silence": True}

    p20 = float(np.percentile(frame_rms, 20))
    p95 = float(np.percentile(frame_rms, 95))
    is_full_silence = bool(p95 < 1e-5)
    if is_full_silence:
        return {"speech_fraction": 0.0, "snr_proxy_db": 0.0, "is_full_silence": True}
    speech_threshold = max(0.01, p20 * 1.8)
    speech_fraction = float(np.mean(frame_rms > speech_threshold))
    snr_proxy_db = float(20.0 * np.log10((p95 + 1e-8) / (p20 + 1e-8)))
    return {"speech_fraction": speech_fraction, "snr_proxy_db": snr_proxy_db, "is_full_silence": False}


def normalize_wav2vec2_text(tokenizer: object, text: str) -> str:
    normalize = getattr(tokenizer, "_normalize", None)
    if callable(normalize):
        return str(normalize(text))
    text = text.upper()
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Wav2Vec2CtcScorer:
    def __init__(self, model_name: str, device: str, dtype: str = "auto") -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        if dtype == "auto" and device.startswith("cuda"):
            torch_dtype = torch.float16
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, dtype=torch_dtype).to(device)
        self.model.eval()
        self.device = device
        self.model_dtype = next(self.model.parameters()).dtype
        self.blank_id = self.model.config.pad_token_id
        if self.blank_id is None:
            self.blank_id = self.processor.tokenizer.pad_token_id
        if self.blank_id is None:
            self.blank_id = 0

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        features = compute_audio_gate_features(waveform, sample_rate=16000)
        return waveform.squeeze(0).detach().cpu().float().numpy(), features

    def ctc_nll(self, audio: np.ndarray, text: str) -> Tuple[float, int, str]:
        normalized_text = normalize_wav2vec2_text(self.processor.tokenizer, str(text))
        labels = self.processor.tokenizer(normalized_text).input_ids
        labels = [int(label) for label in labels if int(label) != self.blank_id]
        token_count = len(labels)
        if token_count == 0:
            return float("nan"), 0, normalized_text

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(device=self.device, dtype=self.model_dtype)
        with torch.inference_mode():
            logits = self.model(input_values).logits.float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long, device=self.device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
            target_lengths = torch.tensor([token_count], dtype=torch.long, device=self.device)
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                label_tensor.unsqueeze(0),
                input_lengths,
                target_lengths,
                blank=int(self.blank_id),
                reduction="sum",
                zero_infinity=True,
            )
        return float(loss.detach().cpu().item()), token_count, normalized_text


def cache_key(row: pd.Series, text_role: str, text: str, model_name: str) -> str:
    payload = "||".join([
        str(row.get("audio_path", "")),
        str(row.get("utterance_id", "")),
        text_role,
        str(text),
        model_name,
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_score_cache(cache_path: Path) -> Dict[str, Dict[str, object]]:
    if not cache_path.exists():
        return {}
    cache: Dict[str, Dict[str, object]] = {}
    with open(cache_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cache[str(item["cache_key"])] = item
    return cache


def append_cache_rows(cache_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def score_rows(
    df: pd.DataFrame,
    scorer: Wav2Vec2CtcScorer,
    model_name: str,
    cache_path: Path,
    limit: Optional[int] = None,
    print_examples: bool = False,
) -> pd.DataFrame:
    cache = load_score_cache(cache_path)
    out = df.copy()
    pending_cache_rows: List[Dict[str, object]] = []
    indices = list(out.index[:limit]) if limit is not None else list(out.index)

    score_columns = [
        "ref_ctc_nll",
        "hyp_ctc_nll",
        "ref_ctc_token_count",
        "hyp_ctc_token_count",
        "grounding_gap",
        "ref_w2v2_text",
        "hyp_w2v2_text",
        "speech_fraction",
        "snr_proxy_db",
        "is_full_silence",
    ]
    for col in score_columns:
        if col not in out.columns:
            out[col] = np.nan

    for ordinal, idx in enumerate(indices, start=1):
        row = out.loc[idx]
        audio_path = str(row["audio_path"])
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio file for row {idx}: {audio_path}")

        audio: Optional[np.ndarray] = None
        audio_features: Optional[Dict[str, float]] = None
        scored = {}
        for role, text in [("ref", row["reference"]), ("hyp", row["hypothesis"] )]:
            key = cache_key(row, role, str(text), model_name)
            cached = cache.get(key)
            if cached is None:
                if audio is None:
                    audio, audio_features = scorer.load_audio(audio_path)
                raw_nll, token_count, normalized_text = scorer.ctc_nll(audio, str(text))
                normalized_nll = raw_nll / token_count if token_count > 0 else float("nan")
                cached = {
                    "cache_key": key,
                    "utterance_id": str(row.get("utterance_id", "")),
                    "audio_path": audio_path,
                    "role": role,
                    "model_name": model_name,
                    "raw_ctc_nll": raw_nll,
                    "ctc_nll": normalized_nll,
                    "ctc_token_count": token_count,
                    "normalized_text": normalized_text,
                }
                cache[key] = cached
                pending_cache_rows.append(cached)
            scored[role] = cached

        if audio_features is None:
            if pd.isna(out.at[idx, "speech_fraction"]):
                audio, audio_features = scorer.load_audio(audio_path)
            else:
                audio_features = {
                    "speech_fraction": out.at[idx, "speech_fraction"],
                    "snr_proxy_db": out.at[idx, "snr_proxy_db"],
                    "is_full_silence": out.at[idx, "is_full_silence"],
                }

        out.at[idx, "ref_ctc_nll"] = float(scored["ref"]["ctc_nll"])
        out.at[idx, "hyp_ctc_nll"] = float(scored["hyp"]["ctc_nll"])
        out.at[idx, "ref_ctc_token_count"] = int(scored["ref"]["ctc_token_count"])
        out.at[idx, "hyp_ctc_token_count"] = int(scored["hyp"]["ctc_token_count"])
        out.at[idx, "grounding_gap"] = float(scored["hyp"]["ctc_nll"]) - float(scored["ref"]["ctc_nll"])
        out.at[idx, "ref_w2v2_text"] = scored["ref"]["normalized_text"]
        out.at[idx, "hyp_w2v2_text"] = scored["hyp"]["normalized_text"]
        out.at[idx, "speech_fraction"] = audio_features["speech_fraction"]
        out.at[idx, "snr_proxy_db"] = audio_features["snr_proxy_db"]
        out.at[idx, "is_full_silence"] = bool(audio_features["is_full_silence"])

        if print_examples:
            print("\n" + "-" * 80)
            print(f"example {ordinal} utterance_id={row.get('utterance_id', '')} condition={row.get('condition', '')}")
            print(f"reference:  {row['reference']}")
            print(f"hypothesis: {row['hypothesis']}")
            print(f"WER: {float(row['WER']):.4f}")
            print(f"ref_ctc_nll: {out.at[idx, 'ref_ctc_nll']:.4f}")
            print(f"hyp_ctc_nll: {out.at[idx, 'hyp_ctc_nll']:.4f}")
            print(f"grounding_gap: {out.at[idx, 'grounding_gap']:.4f}")

        if len(pending_cache_rows) >= 100:
            append_cache_rows(cache_path, pending_cache_rows)
            pending_cache_rows = []
        if ordinal % 25 == 0:
            print(f"Scored {ordinal}/{len(indices)} rows", flush=True)

    append_cache_rows(cache_path, pending_cache_rows)
    return out


def add_threshold_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base = out[out["condition"].eq("base")]
    if base.empty:
        base = out
    mean_wer = float(base["WER"].mean())
    mean_qwen = float(base["Qwen3_score"].mean())
    top25_wer = float(out["WER"].quantile(0.75))
    top25_qwen = float(out["Qwen3_score"].quantile(0.75))

    out["hallucination_baseline_mean_wer_qwen"] = (out["WER"] > mean_wer) & (out["Qwen3_score"] > mean_qwen)
    out["hallucination_top25_wer_qwen"] = (out["WER"] >= top25_wer) & (out["Qwen3_score"] >= top25_qwen)
    out["hallucination_wer_gt_0p5_qwen_gt_0p6"] = (out["WER"] > 0.5) & (out["Qwen3_score"] > 0.6)
    out["hallucination_wer_gt_0p5_qwen_gt_0p7"] = (out["WER"] > 0.5) & (out["Qwen3_score"] > 0.7)
    out["hallucination_existing_or_baseline"] = out["hallucination_baseline_mean_wer_qwen"]
    return out


@dataclass
class MatchResult:
    pairs: pd.DataFrame
    tolerance: float


def match_pairs(
    df: pd.DataFrame,
    flag_col: str,
    base_tolerance: float = 0.05,
    max_tolerance: float = 0.30,
    seed: int = RNG_SEED,
) -> MatchResult:
    rng = random.Random(seed)
    working = df[~df["is_full_silence"].astype(bool)].copy()
    working = working.dropna(subset=["WER", "grounding_gap", "hyp_ctc_nll", flag_col])
    hall = working[working[flag_col].astype(bool)].copy()
    controls = working[~working[flag_col].astype(bool)].copy()
    if hall.empty or controls.empty:
        return MatchResult(pd.DataFrame(), base_tolerance)

    controls_reset = controls.reset_index().rename(columns={"index": "_control_index"})
    control_wer = controls_reset["WER"].astype(float).to_numpy()
    sorted_positions = np.argsort(control_wer)
    sorted_wer = control_wer[sorted_positions]

    for tolerance in [base_tolerance, 0.075, 0.10, 0.15, 0.20, max_tolerance]:
        available = np.ones(len(controls_reset), dtype=bool)
        pairs = []
        hall_indices = hall.index.tolist()
        rng.shuffle(hall_indices)
        for hall_idx in hall_indices:
            hrow = hall.loc[hall_idx]
            low = float(hrow["WER"]) - tolerance
            high = float(hrow["WER"]) + tolerance
            left = int(np.searchsorted(sorted_wer, low, side="left"))
            right = int(np.searchsorted(sorted_wer, high, side="right"))
            candidate_positions = sorted_positions[left:right]
            candidate_positions = candidate_positions[available[candidate_positions]]
            if len(candidate_positions) == 0:
                continue
            same_group = controls_reset.iloc[candidate_positions]
            for group_cols in [["model_name", "condition", "perturbation_condition"], ["condition", "perturbation_condition"], ["condition"], []]:
                candidates = same_group
                for col in group_cols:
                    if col in candidates.columns and col in hrow.index:
                        candidates = candidates[candidates[col].astype(str) == str(hrow[col])]
                if not candidates.empty:
                    break
            if candidates.empty:
                continue
            candidates = candidates.assign(
                wer_delta=(candidates["WER"] - hrow["WER"]).abs(),
                length_delta=(candidates["hyp_ctc_token_count"] - hrow["hyp_ctc_token_count"]).abs(),
            ).sort_values(["wer_delta", "length_delta", "grounding_gap"])
            control_position = int(candidates.index[0])
            available[control_position] = False
            crow = controls_reset.loc[control_position]
            control_idx = crow["_control_index"]
            pairs.append({
                "analysis": flag_col,
                "match_tolerance": tolerance,
                "hallucination_index": hall_idx,
                "control_index": control_idx,
                "hallucination_utterance_id": hrow["utterance_id"],
                "control_utterance_id": crow["utterance_id"],
                "condition": hrow.get("condition", "unknown"),
                "perturbation_condition": hrow.get("perturbation_condition", "unknown"),
                "hallucination_WER": hrow["WER"],
                "control_WER": crow["WER"],
                "hallucination_hyp_ctc_nll": hrow["hyp_ctc_nll"],
                "control_hyp_ctc_nll": crow["hyp_ctc_nll"],
                "hallucination_grounding_gap": hrow["grounding_gap"],
                "control_grounding_gap": crow["grounding_gap"],
                "grounding_gap_diff": hrow["grounding_gap"] - crow["grounding_gap"],
                "hyp_length_delta": abs(float(hrow["hyp_ctc_token_count"]) - float(crow["hyp_ctc_token_count"])),
            })
        if pairs or tolerance >= max_tolerance:
            return MatchResult(pd.DataFrame(pairs), tolerance)
    return MatchResult(pd.DataFrame(), max_tolerance)


def bootstrap_ci(values: np.ndarray, seed: int = RNG_SEED, n_boot: int = 5000) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_pvalue(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if values.size == 0:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, alternative="greater").pvalue)
    except Exception:
        positive = int(np.sum(values > 0))
        n = int(values.size)
        return float(sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n))


def spearman_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr

        result = spearmanr(valid["x"], valid["y"])
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float(valid["x"].rank().corr(valid["y"].rank())), float("nan")


def summarize_pairs(pairs: pd.DataFrame, scored_df: pd.DataFrame, analysis: str, tolerance: float) -> Dict[str, object]:
    if pairs.empty:
        return {"section": "matched_pairs", "analysis": analysis, "N_pairs": 0, "match_tolerance": tolerance}
    diff = pairs["grounding_gap_diff"].astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "section": "matched_pairs",
        "analysis": analysis,
        "N_pairs": len(pairs),
        "match_tolerance": tolerance,
        "hallucination_mean_WER": pairs["hallucination_WER"].mean(),
        "hallucination_median_WER": pairs["hallucination_WER"].median(),
        "control_mean_WER": pairs["control_WER"].mean(),
        "control_median_WER": pairs["control_WER"].median(),
        "hallucination_mean_hyp_ctc_nll": pairs["hallucination_hyp_ctc_nll"].mean(),
        "hallucination_median_hyp_ctc_nll": pairs["hallucination_hyp_ctc_nll"].median(),
        "control_mean_hyp_ctc_nll": pairs["control_hyp_ctc_nll"].mean(),
        "control_median_hyp_ctc_nll": pairs["control_hyp_ctc_nll"].median(),
        "hallucination_mean_grounding_gap": pairs["hallucination_grounding_gap"].mean(),
        "hallucination_median_grounding_gap": pairs["hallucination_grounding_gap"].median(),
        "control_mean_grounding_gap": pairs["control_grounding_gap"].mean(),
        "control_median_grounding_gap": pairs["control_grounding_gap"].median(),
        "mean_grounding_gap_diff": float(np.mean(diff)),
        "median_grounding_gap_diff": float(np.median(diff)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wilcoxon_p_greater": wilcoxon_pvalue(diff),
        "effect_size_mean_diff_over_sd": float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)) if len(diff) > 1 else float("nan"),
    }


def build_summary(scored_df: pd.DataFrame, pair_results: Sequence[MatchResult], analyses: Sequence[str]) -> pd.DataFrame:
    rows = []
    for result, analysis in zip(pair_results, analyses):
        rows.append(summarize_pairs(result.pairs, scored_df, analysis, result.tolerance))

    for x_col, y_col in [
        ("grounding_gap", "WER"),
        ("grounding_gap", "Qwen3_score"),
        ("grounding_gap", "GPT2_score"),
        ("WER", "Qwen3_score"),
        ("WER", "bigram_rep_count"),
        ("WER", "trigram_rep_count"),
        ("WER", "fourgram_rep_count"),
    ]:
        if x_col in scored_df.columns and y_col in scored_df.columns:
            rho, pvalue = spearman_corr(pd.to_numeric(scored_df[x_col], errors="coerce"), pd.to_numeric(scored_df[y_col], errors="coerce"))
            rows.append({"section": "spearman", "analysis": f"{x_col}_vs_{y_col}", "rho": rho, "pvalue": pvalue, "N": scored_df[[x_col, y_col]].dropna().shape[0]})

    for condition, group in scored_df.groupby("condition", sort=False):
        rows.append({
            "section": "condition_grounding_gap",
            "analysis": str(condition),
            "N": len(group),
            "mean_grounding_gap": group["grounding_gap"].mean(),
            "median_grounding_gap": group["grounding_gap"].median(),
            "mean_hyp_ctc_nll": group["hyp_ctc_nll"].mean(),
            "median_hyp_ctc_nll": group["hyp_ctc_nll"].median(),
            "full_silence_rate": group["is_full_silence"].astype(bool).mean(),
        })

    silence = scored_df[scored_df["is_full_silence"].astype(bool)]
    if not silence.empty:
        rows.append({
            "section": "full_silence_sanity",
            "analysis": "full_silence",
            "N": len(silence),
            "mean_grounding_gap": silence["grounding_gap"].mean(),
            "median_grounding_gap": silence["grounding_gap"].median(),
            "mean_hyp_ctc_nll": silence["hyp_ctc_nll"].mean(),
        })
    return pd.DataFrame(rows)


def select_sanity_examples(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    buckets = []
    buckets.append(df.sort_values("WER").head(max(1, limit // 5)))
    if "condition" in df.columns:
        buckets.append(df[df["condition"].eq("ur")].sort_values("Qwen3_score", ascending=False).head(max(1, limit // 5)))
        buckets.append(df[df["condition"].eq("rr")].sort_values("trigram_rep_count", ascending=False).head(max(1, limit // 5)))
    buckets.append(df.sort_values("WER", ascending=False).head(max(1, limit // 5)))
    if "is_full_silence" in df.columns:
        buckets.append(df[df["is_full_silence"].astype(bool)].head(max(1, limit // 5)))
    sample = pd.concat([bucket for bucket in buckets if not bucket.empty]).drop_duplicates(subset=["utterance_id", "condition"])
    if len(sample) < limit:
        sample = pd.concat([sample, df.sample(min(limit - len(sample), len(df)), random_state=RNG_SEED)]).drop_duplicates(subset=["utterance_id", "condition"])
    return sample.head(limit)


def make_plots(scored_df: pd.DataFrame, matched_pairs: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not matched_pairs.empty:
        plt.figure(figsize=(7, 5))
        values = [matched_pairs["control_grounding_gap"].astype(float), matched_pairs["hallucination_grounding_gap"].astype(float)]
        plt.boxplot(values, labels=["WER-matched controls", "Hallucination-like"], showmeans=True)
        plt.ylabel("Grounding gap (hyp CTC NLL - ref CTC NLL)")
        plt.title("Matched Acoustic Grounding")
        plt.tight_layout()
        plt.savefig(output_dir / "matched_grounding_gap.png", dpi=200)
        plt.close()

    plt.figure(figsize=(7, 5))
    for condition, group in scored_df.groupby("condition", sort=False):
        plt.scatter(group["WER"], group["grounding_gap"], s=12, alpha=0.55, label=str(condition))
    plt.xlabel("WER")
    plt.ylabel("Grounding gap")
    plt.title("Grounding Gap vs WER")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "grounding_gap_vs_wer.png", dpi=200)
    plt.close()

    condition_groups = [group["grounding_gap"].dropna().astype(float) for _, group in scored_df.groupby("condition", sort=False)]
    labels = [str(condition) for condition, _ in scored_df.groupby("condition", sort=False)]
    if condition_groups:
        plt.figure(figsize=(7, 5))
        plt.boxplot(condition_groups, labels=labels, showmeans=True)
        plt.xlabel("Condition")
        plt.ylabel("Grounding gap")
        plt.title("Condition-Level Grounding Gap")
        plt.tight_layout()
        plt.savefig(output_dir / "condition_grounding_gap.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wav2vec2 CTC acoustic-grounding validation for saved ASR hypotheses.")
    parser.add_argument("--project_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input_csv", type=Path, default=None, help="Per-utterance metrics CSV with hypotheses and scores.")
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "acoustic_validation")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16"])
    parser.add_argument("--qwen_col", default=None)
    parser.add_argument("--gpt2_col", default=None)
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--max_score_rows", type=int, default=None, help="Limit wav2vec2 scoring for sanity/probe runs.")
    parser.add_argument("--sanity_only", action="store_true", help="Score and print sanity examples without matched statistics.")
    parser.add_argument("--skip_scoring", action="store_true", help="Reuse an existing acoustic_grounding_per_utterance.csv.")
    parser.add_argument("--max_manifest_mismatch_rate", type=float, default=0.25, help="Allowed row-order manifest/reference mismatch rate when reconstructing audio paths.")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache_path or args.output_dir / "wav2vec2_ctc_score_cache.jsonl"
    per_utterance_path = args.output_dir / "acoustic_grounding_per_utterance.csv"
    pairs_path = args.output_dir / "acoustic_grounding_matched_pairs.csv"
    summary_path = args.output_dir / "acoustic_grounding_summary.csv"

    if args.skip_scoring:
        scored_df = add_threshold_flags(pd.read_csv(per_utterance_path))
        scored_df.to_csv(per_utterance_path, index=False)
    else:
        input_csv = args.input_csv or select_input_csv(DEFAULT_INPUT_CANDIDATES, args.qwen_col, args.gpt2_col)
        raw_df = pd.read_csv(input_csv)
        manifest_df = load_manifest_audio_paths(args.test_tsv, args.clips_dir) if args.test_tsv and args.clips_dir else None
        with_audio = attach_audio_paths(raw_df, manifest_df)
        standardized, qwen_col, gpt2_col = standardize_columns(with_audio, args.qwen_col, args.gpt2_col)
        alignment = validate_manifest_alignment(standardized, args.max_manifest_mismatch_rate)
        standardized = add_threshold_flags(standardized)

        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
        scorer = Wav2Vec2CtcScorer(args.model_name, device=device, dtype=args.dtype)
        if args.sanity_only:
            sanity_df = select_sanity_examples(standardized, limit=args.max_score_rows or 20)
            scored_df = score_rows(sanity_df, scorer, args.model_name, cache_path, limit=args.max_score_rows, print_examples=True)
        else:
            scored_df = score_rows(standardized, scorer, args.model_name, cache_path, limit=args.max_score_rows)
        scored_df.to_csv(per_utterance_path, index=False)
        metadata = {
            "input_csv": str(input_csv),
            "test_tsv": str(args.test_tsv),
            "clips_dir": str(args.clips_dir),
            "wav2vec2_model": args.model_name,
            "qwen_source_column": qwen_col,
            "gpt2_source_column": gpt2_col,
            "manifest_alignment": alignment,
            "seed": args.seed,
        }
        with open(args.output_dir / "acoustic_grounding_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    if args.sanity_only:
        print(f"\nSaved sanity scores to {per_utterance_path}")
        print(f"Cache: {cache_path}")
        return

    analyses = [
        "hallucination_existing_or_baseline",
        "hallucination_baseline_mean_wer_qwen",
        "hallucination_top25_wer_qwen",
        "hallucination_wer_gt_0p5_qwen_gt_0p6",
        "hallucination_wer_gt_0p5_qwen_gt_0p7",
    ]
    available_analyses = [col for col in analyses if col in scored_df.columns]
    pair_results = [match_pairs(scored_df, col, seed=args.seed) for col in available_analyses]
    all_pairs = pd.concat([result.pairs for result in pair_results if not result.pairs.empty], ignore_index=True) if pair_results else pd.DataFrame()
    all_pairs.to_csv(pairs_path, index=False)
    summary_df = build_summary(scored_df, pair_results, available_analyses)
    summary_df.to_csv(summary_path, index=False)
    primary_pairs = pair_results[0].pairs if pair_results and not pair_results[0].pairs.empty else all_pairs
    make_plots(scored_df, primary_pairs, args.output_dir)

    print(f"Saved per-utterance acoustic scores to {per_utterance_path}")
    print(f"Saved matched pairs to {pairs_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    main()