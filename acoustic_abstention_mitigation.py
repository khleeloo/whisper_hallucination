#!/usr/bin/env python3
"""Hallucination mitigation experiment: acoustic-consistency abstention.

This experiment is deliberately a *pipeline validation*, not a claim of a novel
mitigation algorithm. It uses an independent wav2vec2-CTC model as a
reference-free acoustic-support gate:

    accept Whisper hypothesis iff normalized CTC NLL(x, hypothesis) <= tau

Protocol
--------
1. Generate Base checkpoint-10000 outputs on a disjoint DEV split under:
   clean, full-noise 0.5, and full-noise 0.75.
2. Label hallucination-like DEV outputs with the frozen matched-Base
   WER + GPT-2 plausibility criterion.
3. Select one global CTC-support threshold tau on DEV. Among thresholds that
   retain at least --min_clean_coverage of clean DEV outputs, choose the one
   that rejects the largest fraction of hallucination-like stressed outputs.
4. Freeze tau.
5. Apply exactly the same gate to held-out TEST clean and stressed outputs.
6. Report coverage, emitted hallucination incidence, residual hallucination
   rate among accepted outputs, hallucination capture recall, and accepted WER.

Important:
- The gate itself never uses the reference transcription or an LM score at test
  time. References/LM scores are used only for evaluation labels.
- The same deterministic perturbation realization is reconstructed for Whisper
  decoding and wav2vec2 scoring. This avoids the stochastic-noise mismatch in
  legacy stress files.
- wav2vec2 is the mitigation signal, so grounding-gap improvement is *not* used
  as a mitigation outcome (that would be circular).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

from acoustic_grounding_validation import (
    DEFAULT_MODEL_NAME as DEFAULT_WAV2VEC2_MODEL,
    Wav2Vec2CtcScorer,
    normalize_wav2vec2_text,
)
from mitigation_experiment import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GPT2_MODEL,
    _default_lm_plausibility,
    _load_whisper_model,
    parse_perturbation,
)
from evaluate_whisper_validation import compute_wer_metrics

SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DATA_ROOT = Path("/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination")
CV_ROOT = Path("/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en")

DEFAULT_TEST_TSV = DATA_ROOT / "test.tsv"
DEFAULT_CLIPS_DIR = CV_ROOT / "clips"
DEFAULT_BASE_MODEL_DIR = SCRATCH_ROOT / "base" / "checkpoint-10000"
DEFAULT_OUTPUT_DIR = SCRATCH_ROOT / "hallucination_mitigation_acoustic"

# Frozen from the checkpoint-matched Base clean 1000-utterance evaluation.
DEFAULT_HALL_WER_THRESHOLD = 0.116215
DEFAULT_HALL_PLAUS_THRESHOLD = 0.892170

DEFAULT_PERTURBATIONS = [
    "none",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]

DEV_TSV_CANDIDATES = [
    DATA_ROOT / "dev.tsv",
    DATA_ROOT / "validation.tsv",
    CV_ROOT / "dev.tsv",
]

GENERATED_COLUMNS = [
    "split",
    "utterance_id",
    "audio_path",
    "reference",
    "perturbation",
    "hypothesis",
]


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "||".join([str(base_seed), *[str(x) for x in parts]])
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def resolve_dev_tsv(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"DEV TSV does not exist: {explicit}")
        return explicit
    for candidate in DEV_TSV_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = "\n  ".join(str(p) for p in DEV_TSV_CANDIDATES)
    raise FileNotFoundError(
        "Could not auto-detect a disjoint DEV TSV. Checked:\n  "
        + checked
        + "\nPass --dev_tsv /path/to/dev.tsv (or DEV_TSV=... in the sbatch command)."
    )


def load_manifest(
    tsv_path: Path,
    clips_dir: Path,
    *,
    split: str,
    max_samples: Optional[int],
) -> pd.DataFrame:
    """Reproduce legacy max_samples semantics: truncate TSV rows before file filtering."""
    rows: List[Dict[str, object]] = []
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "path" not in reader.fieldnames or "sentence" not in reader.fieldnames:
            raise ValueError(f"Expected Common Voice columns 'path' and 'sentence' in {tsv_path}")
        for idx, row in enumerate(reader):
            if max_samples is not None and idx >= max_samples:
                break
            rel_path = str(row["path"])
            audio_path = clips_dir / rel_path
            if not audio_path.exists():
                continue
            rows.append(
                {
                    "split": split,
                    "manifest_index": idx,
                    "utterance_id": Path(rel_path).stem.lower(),
                    "audio_path": str(audio_path),
                    "reference": str(row["sentence"]),
                }
            )
    if not rows:
        raise ValueError(f"No usable rows loaded from {tsv_path}")
    return pd.DataFrame(rows)


def validate_disjoint(dev: pd.DataFrame, test: pd.DataFrame, *, allow_overlap: bool) -> None:
    overlap = sorted(set(dev["utterance_id"]) & set(test["utterance_id"]))
    if overlap and not allow_overlap:
        raise ValueError(
            f"DEV and TEST overlap by {len(overlap)} utterance IDs. "
            f"Examples: {overlap[:10]}. Use a disjoint DEV split; "
            "--allow_overlap is only for debugging."
        )
    if overlap:
        print(f"WARNING: DEV/TEST overlap allowed for debugging: {len(overlap)} IDs", flush=True)


def deterministic_perturb(
    waveform: torch.Tensor,
    sample_rate: int,
    label: str,
    *,
    seed: int,
) -> torch.Tensor:
    """Mirror project perturbations but make stochastic noise exactly reproducible."""
    p = parse_perturbation(label)
    perturb_type = p.perturb_type
    amplitude = float(p.amplitude)
    duration = float(p.duration)

    if perturb_type == "none":
        out = waveform
    elif perturb_type == "onset_noise":
        n_samples = min(int(duration * sample_rate), waveform.shape[-1])
        gen = torch.Generator(device=waveform.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            waveform[..., :n_samples].shape,
            dtype=waveform.dtype,
            device=waveform.device,
            generator=gen,
        ) * amplitude
        out = waveform.clone()
        out[..., :n_samples] = out[..., :n_samples] + noise
    elif perturb_type == "full_noise":
        gen = torch.Generator(device=waveform.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            waveform.shape,
            dtype=waveform.dtype,
            device=waveform.device,
            generator=gen,
        ) * amplitude
        out = waveform + noise
    elif perturb_type == "reverb":
        strength = max(0.0, amplitude)
        delay_samples = max(1, int(0.035 * sample_rate))
        tail_seconds = duration if duration > 0 else 0.45
        tail_samples = max(delay_samples + 1, int(tail_seconds * sample_rate))
        impulse = torch.zeros(
            1, 1, tail_samples, dtype=waveform.dtype, device=waveform.device
        )
        impulse[..., 0] = 1.0
        tap = delay_samples
        tap_idx = 1
        while tap < tail_samples:
            impulse[..., tap] = strength * (0.62 ** tap_idx)
            tap_idx += 1
            tap += delay_samples
        original_shape = waveform.shape
        convolved = torch.nn.functional.conv1d(
            waveform.reshape(-1, 1, waveform.shape[-1]),
            impulse,
            padding=tail_samples - 1,
        )[..., : waveform.shape[-1]]
        out = convolved.reshape(original_shape)
    elif perturb_type == "silence":
        out = torch.zeros_like(waveform)
    elif perturb_type == "leading_silence":
        n_samples = max(0, int(duration * sample_rate))
        silence = torch.zeros(
            waveform.shape[:-1] + (n_samples,),
            dtype=waveform.dtype,
            device=waveform.device,
        )
        out = torch.cat([silence, waveform], dim=-1)
    elif perturb_type == "speech_band_noise":
        gen = torch.Generator(device=waveform.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            waveform.shape,
            dtype=waveform.dtype,
            device=waveform.device,
            generator=gen,
        )
        spectrum = torch.fft.rfft(noise, dim=-1)
        freqs = torch.fft.rfftfreq(noise.shape[-1], d=1.0 / sample_rate).to(noise.device)
        mask = ((freqs >= 300.0) & (freqs <= 3400.0)).to(spectrum.dtype)
        filtered = torch.fft.irfft(spectrum * mask, n=noise.shape[-1], dim=-1)
        filtered = filtered / (filtered.pow(2).mean().sqrt() + 1e-8)
        signal_rms = waveform.pow(2).mean().sqrt().clamp_min(1e-4)
        out = waveform + filtered * signal_rms * amplitude
    else:
        raise ValueError(f"Unsupported perturbation: {label}")

    return torch.clamp(out, -1.0, 1.0)


def load_exact_waveform(row: object, label: str, *, base_seed: int) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(str(row.audio_path))
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    seed = stable_seed(base_seed, row.split, row.utterance_id, label)
    waveform = deterministic_perturb(waveform, 16000, label, seed=seed)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).detach().cpu().float().numpy()


def decode_manifest(
    model: object,
    processor: object,
    manifest: pd.DataFrame,
    perturbation: str,
    *,
    device: str,
    batch_size: int,
    base_seed: int,
) -> pd.DataFrame:
    hypotheses: List[str] = []
    records = list(manifest.itertuples(index=False))
    model_dtype = next(model.parameters()).dtype

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        waveforms = [
            load_exact_waveform(row, perturbation, base_seed=base_seed)
            for row in batch
        ]
        inputs = processor.feature_extractor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(device)
        inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)

        with torch.inference_mode():
            ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=225,
                language="en",
                task="transcribe",
            )
        hypotheses.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))

        done = min(start + len(batch), len(records))
        if start == 0 or done == len(records) or (start // batch_size) % 10 == 0:
            print(
                f"  {manifest.iloc[0]['split']} {perturbation}: "
                f"transcribed {done}/{len(records)}",
                flush=True,
            )

    out = manifest[["split", "utterance_id", "audio_path", "reference"]].copy()
    out["perturbation"] = perturbation
    out["hypothesis"] = hypotheses
    return out[GENERATED_COLUMNS]


def add_evaluation_labels(
    df: pd.DataFrame,
    *,
    device: str,
    gpt2_model: str,
    lm_batch_size: int,
    wer_threshold: float,
    plaus_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    wer_rows = compute_wer_metrics(
        out["hypothesis"].tolist(),
        out["reference"].tolist(),
    )
    out["WER"] = [float(row["wer"]) for row in wer_rows]
    print(f"Scoring {len(out):,} hypotheses with {gpt2_model}...", flush=True)
    out["gpt2_plaus"] = list(
        _default_lm_plausibility(
            out["hypothesis"].tolist(),
            out["reference"].tolist(),
            gpt2_model,
            device,
            lm_batch_size,
        )
    )
    out["hallucination_like"] = (
        (out["WER"].astype(float) > float(wer_threshold))
        & (out["gpt2_plaus"].astype(float) > float(plaus_threshold))
    )
    return out


def ctc_cache_key(row: object, model_name: str, base_seed: int) -> str:
    payload = "||".join(
        [
            str(base_seed),
            str(row.split),
            str(row.utterance_id),
            str(row.perturbation),
            str(row.hypothesis),
            model_name,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_ctc_cache(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    cache: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[str(item["key"])] = float(item["normalized_ctc_nll"])
    return cache


def append_ctc_cache(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalized_ctc_nll_batch(
    scorer: Wav2Vec2CtcScorer,
    audios: Sequence[np.ndarray],
    texts: Sequence[str],
) -> List[float]:
    """Per-utterance CTC NLL/token. Impossible alignments remain +inf."""
    normalized_texts = [
        normalize_wav2vec2_text(scorer.processor.tokenizer, str(text))
        for text in texts
    ]
    label_lists: List[List[int]] = []
    for text in normalized_texts:
        ids = scorer.processor.tokenizer(text).input_ids
        ids = [int(x) for x in ids if int(x) != int(scorer.blank_id)]
        label_lists.append(ids)

    scores = [float("inf")] * len(texts)
    nonempty = [i for i, ids in enumerate(label_lists) if ids]
    if not nonempty:
        return scores

    subset_audio = [audios[i] for i in nonempty]
    inputs = scorer.processor(
        subset_audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(
        device=scorer.device,
        dtype=scorer.model_dtype,
    )
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is None:
        raw_lengths = torch.tensor(
            [len(a) for a in subset_audio],
            dtype=torch.long,
            device=scorer.device,
        )
    else:
        raw_lengths = attention_mask.sum(-1).to(
            device=scorer.device, dtype=torch.long
        )

    with torch.inference_mode():
        logits = scorer.model(input_values).logits.float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

    get_lengths = getattr(scorer.model, "_get_feat_extract_output_lengths", None)
    if callable(get_lengths):
        input_lengths = get_lengths(raw_lengths).to(dtype=torch.long)
    else:
        max_raw = int(raw_lengths.max().item())
        input_lengths = torch.clamp(
            torch.round(raw_lengths.float() / max_raw * log_probs.shape[0]).long(),
            min=1,
            max=log_probs.shape[0],
        )

    target_lengths = torch.tensor(
        [len(label_lists[i]) for i in nonempty],
        dtype=torch.long,
        device=scorer.device,
    )
    max_target = int(target_lengths.max().item())
    targets = torch.full(
        (len(nonempty), max_target),
        int(scorer.blank_id),
        dtype=torch.long,
        device=scorer.device,
    )
    for j, original_idx in enumerate(nonempty):
        ids = torch.tensor(
            label_lists[original_idx],
            dtype=torch.long,
            device=scorer.device,
        )
        targets[j, : len(ids)] = ids

    losses = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=int(scorer.blank_id),
        reduction="none",
        zero_infinity=False,
    )
    normalized = losses / target_lengths.float()
    for j, original_idx in enumerate(nonempty):
        value = float(normalized[j].detach().cpu().item())
        scores[original_idx] = value if math.isfinite(value) else float("inf")
    return scores


def score_ctc_support(
    df: pd.DataFrame,
    *,
    wav2vec2_model: str,
    device: str,
    cache_path: Path,
    batch_size: int,
    base_seed: int,
) -> pd.DataFrame:
    out = df.copy()
    cache = load_ctc_cache(cache_path)
    scorer = Wav2Vec2CtcScorer(wav2vec2_model, device=device)
    scores = np.full(len(out), np.nan, dtype=float)

    rows = list(out.itertuples(index=False))
    pending: List[int] = []
    keys: List[str] = []
    for idx, row in enumerate(rows):
        key = ctc_cache_key(row, wav2vec2_model, base_seed)
        keys.append(key)
        if key in cache:
            scores[idx] = cache[key]
        else:
            pending.append(idx)

    print(
        f"wav2vec2 support: {len(out)-len(pending):,} cached, "
        f"{len(pending):,} to score",
        flush=True,
    )

    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        batch_rows = [rows[i] for i in indices]
        audios = [
            load_exact_waveform(row, row.perturbation, base_seed=base_seed)
            for row in batch_rows
        ]
        batch_scores = normalized_ctc_nll_batch(
            scorer,
            audios,
            [row.hypothesis for row in batch_rows],
        )
        cache_rows: List[Dict[str, object]] = []
        for idx, row, score in zip(indices, batch_rows, batch_scores):
            scores[idx] = score
            cache_rows.append(
                {
                    "key": keys[idx],
                    "split": row.split,
                    "utterance_id": row.utterance_id,
                    "perturbation": row.perturbation,
                    "normalized_ctc_nll": score,
                    "model_name": wav2vec2_model,
                    "seed": base_seed,
                }
            )
        append_ctc_cache(cache_path, cache_rows)
        done = min(start + len(indices), len(pending))
        if start == 0 or done == len(pending) or (start // batch_size) % 10 == 0:
            print(f"  wav2vec2 scored {done}/{len(pending)} uncached rows", flush=True)

    out["ctc_support_nll"] = scores
    del scorer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def accepted_mask(scores: Sequence[float], threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    return np.isfinite(values) & (values <= float(threshold))


def select_gate_threshold(
    dev: pd.DataFrame,
    *,
    min_clean_coverage: float,
) -> Tuple[float, pd.DataFrame]:
    clean = dev[dev["perturbation"] == "none"].copy()
    stress = dev[dev["perturbation"] != "none"].copy()
    if clean.empty or stress.empty:
        raise ValueError("DEV must contain clean and at least one stressed perturbation")

    finite_clean = np.sort(
        clean.loc[np.isfinite(clean["ctc_support_nll"]), "ctc_support_nll"]
        .astype(float)
        .unique()
    )
    if len(finite_clean) == 0:
        raise ValueError("No finite clean DEV CTC scores; cannot select gate threshold")

    records: List[Dict[str, object]] = []
    stress_hall = stress["hallucination_like"].astype(bool).to_numpy()
    n_stress_hall = int(stress_hall.sum())

    for tau in finite_clean:
        clean_accept = accepted_mask(clean["ctc_support_nll"], tau)
        clean_coverage = float(clean_accept.mean())
        if clean_coverage + 1e-12 < min_clean_coverage:
            continue

        stress_accept = accepted_mask(stress["ctc_support_nll"], tau)
        rejected_hall = int((~stress_accept & stress_hall).sum())
        accepted_hall = int((stress_accept & stress_hall).sum())
        records.append(
            {
                "threshold": float(tau),
                "clean_coverage": clean_coverage,
                "clean_rejection_rate": 1.0 - clean_coverage,
                "stress_coverage": float(stress_accept.mean()),
                "stress_hallucinations": n_stress_hall,
                "stress_hallucination_capture_recall": (
                    rejected_hall / n_stress_hall if n_stress_hall else 0.0
                ),
                "stress_emitted_hallucination_incidence": float(
                    accepted_hall / len(stress)
                ),
                "stress_residual_hallucination_among_accepted": (
                    accepted_hall / int(stress_accept.sum())
                    if int(stress_accept.sum())
                    else 0.0
                ),
            }
        )

    if not records:
        raise ValueError(
            f"No threshold satisfies min clean coverage={min_clean_coverage:.3f}. "
            "This can happen if too many clean CTC scores are non-finite."
        )

    table = pd.DataFrame(records)
    table = table.sort_values(
        [
            "stress_hallucination_capture_recall",
            "stress_coverage",
            "clean_coverage",
            "threshold",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return float(table.iloc[0]["threshold"]), table


def bootstrap_delta_incidence(
    hall: np.ndarray,
    accepted: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    hall = np.asarray(hall, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    per_row_delta = (hall & accepted).astype(float) - hall.astype(float)
    point = float(per_row_delta.mean())
    if len(hall) == 0 or n_boot <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(hall), size=len(hall))
        boot[i] = per_row_delta[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize_group(
    group: pd.DataFrame,
    *,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    hall = group["hallucination_like"].astype(bool).to_numpy()
    accept = accepted_mask(group["ctc_support_nll"], threshold)
    emitted_hall = hall & accept
    n_accept = int(accept.sum())
    n_hall = int(hall.sum())
    delta, ci_lo, ci_hi = bootstrap_delta_incidence(
        hall, accept, n_boot=n_boot, seed=seed
    )
    wer = group["WER"].astype(float).to_numpy()
    return {
        "split": str(group.iloc[0]["split"]),
        "perturbation": str(group.iloc[0]["perturbation"]),
        "N": int(len(group)),
        "threshold": float(threshold),
        "hallucination_rate_before": float(hall.mean()),
        "coverage": float(accept.mean()),
        "abstention_rate": float((~accept).mean()),
        "emitted_hallucination_incidence": float(emitted_hall.mean()),
        "residual_hallucination_among_accepted": (
            float(emitted_hall.sum() / n_accept) if n_accept else 0.0
        ),
        "hallucination_capture_recall": (
            float((hall & ~accept).sum() / n_hall) if n_hall else 0.0
        ),
        "mean_WER_all": float(np.mean(wer)),
        "mean_WER_accepted": float(np.mean(wer[accept])) if n_accept else float("nan"),
        "delta_emitted_hallucination_incidence": delta,
        "delta_CI_low": ci_lo,
        "delta_CI_high": ci_hi,
    }


def add_gate_column(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["accepted"] = accepted_mask(out["ctc_support_nll"], threshold)
    out["abstained"] = ~out["accepted"]
    out["emitted_hallucination"] = (
        out["hallucination_like"].astype(bool) & out["accepted"]
    )
    return out


def generate_all(
    dev_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    model_dir: Path,
    base_model_name: str,
    perturbations: Sequence[str],
    device: str,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    print(f"Loading Whisper Base from {model_dir}", flush=True)
    model, base_model, processor = _load_whisper_model(model_dir, base_model_name, device)
    frames: List[pd.DataFrame] = []
    try:
        for manifest in [dev_manifest, test_manifest]:
            for perturbation in perturbations:
                frames.append(
                    decode_manifest(
                        model,
                        processor,
                        manifest,
                        perturbation,
                        device=device,
                        batch_size=batch_size,
                        base_seed=seed,
                    )
                )
    finally:
        model.cpu()
        del model, base_model, processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic-consistency abstention experiment for hallucination-like ASR failures."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dev_tsv", type=Path, default=None)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--base_model_dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--gpt2_model", default=DEFAULT_GPT2_MODEL)
    parser.add_argument("--wav2vec2_model", default=DEFAULT_WAV2VEC2_MODEL)
    parser.add_argument("--perturbations", nargs="+", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--dev_max_samples", type=int, default=1000)
    parser.add_argument("--test_max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ctc_batch_size", type=int, default=8)
    parser.add_argument("--lm_batch_size", type=int, default=8)
    parser.add_argument("--min_clean_coverage", type=float, default=0.98)
    parser.add_argument("--hall_wer_threshold", type=float, default=DEFAULT_HALL_WER_THRESHOLD)
    parser.add_argument("--hall_plaus_threshold", type=float, default=DEFAULT_HALL_PLAUS_THRESHOLD)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse_generated_outputs", action="store_true")
    parser.add_argument("--allow_overlap", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.min_clean_coverage <= 1.0):
        raise ValueError("--min_clean_coverage must be in (0, 1]")
    if "none" not in args.perturbations:
        raise ValueError("--perturbations must include 'none'")
    if not any(label != "none" for label in args.perturbations):
        raise ValueError("--perturbations must include at least one stressed condition")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.output_dir / "generated_outputs.csv"
    scored_path = args.output_dir / "scored_outputs.csv"
    ctc_cache_path = args.output_dir / "ctc_support_cache.jsonl"
    threshold_search_path = args.output_dir / "dev_threshold_search.csv"
    threshold_json_path = args.output_dir / "frozen_gate_threshold.json"
    test_outputs_path = args.output_dir / "test_outputs_with_gate.csv"
    test_summary_path = args.output_dir / "test_mitigation_summary.csv"
    dev_summary_path = args.output_dir / "dev_mitigation_summary.csv"
    report_path = args.output_dir / "report.json"

    dev_tsv = resolve_dev_tsv(args.dev_tsv)
    print("=== Acoustic-consistency abstention experiment ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"DEV TSV: {dev_tsv}", flush=True)
    print(f"TEST TSV: {args.test_tsv}", flush=True)
    print(f"Model: {args.base_model_dir}", flush=True)
    print(f"Perturbations: {args.perturbations}", flush=True)
    print(
        "Frozen hallucination proxy: "
        f"WER>{args.hall_wer_threshold:.6f}, "
        f"GPT2>{args.hall_plaus_threshold:.6f}",
        flush=True,
    )
    print(f"Minimum clean DEV coverage: {args.min_clean_coverage:.3f}", flush=True)

    dev_manifest = load_manifest(
        dev_tsv,
        args.clips_dir,
        split="dev",
        max_samples=args.dev_max_samples,
    )
    test_manifest = load_manifest(
        args.test_tsv,
        args.clips_dir,
        split="test",
        max_samples=args.test_max_samples,
    )
    validate_disjoint(dev_manifest, test_manifest, allow_overlap=args.allow_overlap)
    print(f"DEV rows: {len(dev_manifest):,}; TEST rows: {len(test_manifest):,}", flush=True)

    if args.reuse_generated_outputs and generated_path.exists():
        generated = pd.read_csv(generated_path)
        required = set(GENERATED_COLUMNS)
        if not required.issubset(generated.columns):
            raise ValueError(
                f"{generated_path} lacks required columns: {sorted(required - set(generated.columns))}"
            )
        print(f"Reusing generated outputs: {generated_path}", flush=True)
    else:
        generated = generate_all(
            dev_manifest,
            test_manifest,
            model_dir=args.base_model_dir,
            base_model_name=args.base_model_name,
            perturbations=args.perturbations,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        generated.to_csv(generated_path, index=False)
        print(f"Saved generated outputs: {generated_path}", flush=True)

    expected = {
        (split, perturbation)
        for split in ["dev", "test"]
        for perturbation in args.perturbations
    }
    present = set(zip(generated["split"], generated["perturbation"]))
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Generated outputs missing split/perturbation groups: {missing}")

    scored = add_evaluation_labels(
        generated,
        device=device,
        gpt2_model=args.gpt2_model,
        lm_batch_size=args.lm_batch_size,
        wer_threshold=args.hall_wer_threshold,
        plaus_threshold=args.hall_plaus_threshold,
    )
    scored = score_ctc_support(
        scored,
        wav2vec2_model=args.wav2vec2_model,
        device=device,
        cache_path=ctc_cache_path,
        batch_size=args.ctc_batch_size,
        base_seed=args.seed,
    )
    scored.to_csv(scored_path, index=False)
    print(f"Saved scored outputs: {scored_path}", flush=True)

    dev = scored[scored["split"] == "dev"].copy()
    test = scored[scored["split"] == "test"].copy()

    threshold, search_table = select_gate_threshold(
        dev,
        min_clean_coverage=args.min_clean_coverage,
    )
    search_table.to_csv(threshold_search_path, index=False)

    selected = search_table.iloc[0].to_dict()
    threshold_payload = {
        "ctc_model": args.wav2vec2_model,
        "selection_split": "dev",
        "selection_perturbations": [
            x for x in args.perturbations if x != "none"
        ],
        "threshold": threshold,
        "min_clean_coverage": args.min_clean_coverage,
        "selected_dev_metrics": selected,
        "hallucination_label_for_tuning_only": {
            "criterion": "WER > threshold AND normalized GPT2 plausibility > threshold",
            "wer_threshold": args.hall_wer_threshold,
            "gpt2_plausibility_threshold": args.hall_plaus_threshold,
        },
        "gate_at_test_time": "accept iff finite normalized wav2vec2 CTC hypothesis NLL <= threshold",
        "seed": args.seed,
        "dev_tsv": str(dev_tsv),
        "test_tsv": str(args.test_tsv),
        "base_model_dir": str(args.base_model_dir),
    }
    threshold_json_path.write_text(
        json.dumps(threshold_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen CTC threshold tau={threshold:.6f}", flush=True)
    print(
        "DEV selection: "
        f"clean coverage={selected['clean_coverage']:.3f}, "
        f"stress hall capture={selected['stress_hallucination_capture_recall']:.3f}, "
        f"stress coverage={selected['stress_coverage']:.3f}",
        flush=True,
    )

    dev_gated = add_gate_column(dev, threshold)
    test_gated = add_gate_column(test, threshold)
    test_gated.to_csv(test_outputs_path, index=False)

    dev_rows = []
    for perturbation, group in dev_gated.groupby("perturbation", sort=False):
        dev_rows.append(
            summarize_group(
                group,
                threshold=threshold,
                n_boot=args.bootstrap,
                seed=stable_seed(args.seed, "dev", perturbation, "bootstrap"),
            )
        )
    test_rows = []
    for perturbation, group in test_gated.groupby("perturbation", sort=False):
        test_rows.append(
            summarize_group(
                group,
                threshold=threshold,
                n_boot=args.bootstrap,
                seed=stable_seed(args.seed, "test", perturbation, "bootstrap"),
            )
        )

    dev_summary = pd.DataFrame(dev_rows)
    test_summary = pd.DataFrame(test_rows)
    dev_summary.to_csv(dev_summary_path, index=False)
    test_summary.to_csv(test_summary_path, index=False)

    print("\n=== HELD-OUT TEST mitigation summary ===", flush=True)
    print(
        test_summary[
            [
                "perturbation",
                "N",
                "hallucination_rate_before",
                "coverage",
                "emitted_hallucination_incidence",
                "residual_hallucination_among_accepted",
                "hallucination_capture_recall",
                "mean_WER_accepted",
                "delta_emitted_hallucination_incidence",
                "delta_CI_low",
                "delta_CI_high",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        flush=True,
    )

    report = {
        "experiment": "acoustic_consistency_abstention",
        "purpose": (
            "Proof-of-concept hallucination mitigation inside the diagnostic evaluation pipeline: "
            "tune under magnified acoustic stress, freeze, and quantify held-out clean cost."
        ),
        "threshold": threshold_payload,
        "dev_summary_csv": str(dev_summary_path),
        "test_summary_csv": str(test_summary_path),
        "test_outputs_csv": str(test_outputs_path),
        "generated_outputs_csv": str(generated_path),
        "scored_outputs_csv": str(scored_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs:", flush=True)
    for path in [
        threshold_json_path,
        test_summary_path,
        dev_summary_path,
        test_outputs_path,
        report_path,
    ]:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
