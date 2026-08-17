"""
Evaluate Whisper models with dual-metric framework:
  1. Lexical accuracy (WAcc = 1 - WER)
  2. Sequence plausibility (LM-based sentence probability)
  3. Hallucination-like outputs (above-average plausibility + below-average WAcc)
  4. Repetition analysis (n-gram repetition counts)

Also supports perturbation-based evaluation (noise injection).

Usage:
    # Standard evaluation
    python evaluate_dual_metric.py \
        --model_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/base/final \
        --test_tsv /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv \
        --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results/base \
        --config_name base

    # With perturbation
    python evaluate_dual_metric.py \
        --model_dir ... \
        --perturb_type onset_noise \
        --perturb_amplitude 0.1 \
        --perturb_duration 0.5
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
from collections import Counter

import evaluate
import numpy as np
import torch
import torchaudio
from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


normalizer = BasicTextNormalizer()

DEFAULT_GATE_THRESHOLDS = {
    "T_speech_fraction": 0.12,
    "T_snr_proxy_db": 8.0,
    "T_audio_text_min_words": 6,
    "T_token_path_distance": 0.25,
    "T_beam_logprob_margin": 0.25,
}


def compute_avg_logprobs_from_generate(gen_out, model=None):
    """Compute mean generated-token log probability for each sequence."""
    sequences = getattr(gen_out, "sequences", None)
    scores = list(getattr(gen_out, "scores", []) or [])
    if sequences is None:
        return []
    sequence_scores = getattr(gen_out, "sequences_scores", None)
    if sequence_scores is not None:
        return sequence_scores.detach().float().cpu().tolist()
    if not scores:
        return [0.0] * int(sequences.shape[0])

    if model is not None and hasattr(model, "compute_transition_scores"):
        beam_indices = getattr(gen_out, "beam_indices", None)
        try:
            transition_scores = model.compute_transition_scores(
                sequences,
                tuple(scores),
                beam_indices=beam_indices,
                normalize_logits=True,
            )
            return transition_scores.mean(dim=1).detach().cpu().tolist()
        except Exception:
            pass

    num_steps = min(len(scores), int(sequences.shape[1]))
    if num_steps <= 0:
        return [0.0] * int(sequences.shape[0])

    generated_tokens = sequences[:, -num_steps:].detach().cpu()
    token_logprobs = []
    for step_idx, step_scores in enumerate(scores[-num_steps:]):
        step_logprobs = torch.nn.functional.log_softmax(step_scores.detach().float().cpu(), dim=-1)
        step_tokens = generated_tokens[:, step_idx].clamp(0, step_logprobs.shape[1] - 1).unsqueeze(1)
        token_logprobs.append(step_logprobs.gather(1, step_tokens).squeeze(1))

    stacked = torch.stack(token_logprobs, dim=1)
    return stacked.mean(dim=1).tolist()


def compute_compression_ratio(text):
    """Return len(utf8 text) / len(gzip-compressed utf8 text)."""
    text_bytes = str(text).encode("utf-8")
    if not text_bytes:
        return 0.0
    return len(text_bytes) / len(gzip.compress(text_bytes))


def calibrate_gate_thresholds(avg_logprobs, compression_ratios):
    """Calibrate decoder-only gate thresholds from finite current-run signals."""
    finite_logprobs = [float(x) for x in avg_logprobs if np.isfinite(x)]
    finite_ratios = [float(x) for x in compression_ratios if np.isfinite(x)]
    return {
        "T_logprob": float(np.percentile(finite_logprobs, 5)) if finite_logprobs else 0.0,
        "T_compression": float(np.percentile(finite_ratios, 95)) if finite_ratios else float("inf"),
        **DEFAULT_GATE_THRESHOLDS,
    }


def with_gate_threshold_defaults(thresholds):
    """Return gate thresholds with acoustic defaults filled in for old JSON files."""
    merged = dict(DEFAULT_GATE_THRESHOLDS)
    supplied = dict(thresholds or {})
    if "T_token_path_distance" not in supplied and "T_hypothesis_disagreement" in supplied:
        supplied["T_token_path_distance"] = supplied["T_hypothesis_disagreement"]
    if "T_beam_logprob_margin" not in supplied and "T_logprob_margin" in supplied:
        supplied["T_beam_logprob_margin"] = supplied["T_logprob_margin"]
    merged.update(supplied)
    merged["T_logprob"] = float(merged["T_logprob"])
    merged["T_compression"] = float(merged["T_compression"])
    merged["T_speech_fraction"] = float(merged["T_speech_fraction"])
    merged["T_snr_proxy_db"] = float(merged["T_snr_proxy_db"])
    merged["T_token_path_distance"] = float(merged["T_token_path_distance"])
    merged["T_beam_logprob_margin"] = float(merged["T_beam_logprob_margin"])
    merged["T_audio_text_min_words"] = int(merged["T_audio_text_min_words"])
    return merged


def compute_audio_gate_features(waveform, sample_rate=16000):
    """Compute cheap reference-free acoustic cues from a mono waveform."""
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    samples = waveform.detach().float().reshape(-1).cpu()
    if samples.numel() == 0:
        return {"speech_fraction": 0.0, "snr_proxy_db": 0.0}

    frame_length = max(1, int(0.03 * sample_rate))
    hop_length = max(1, int(0.01 * sample_rate))
    if samples.numel() < frame_length:
        frames = samples.unsqueeze(0)
    else:
        frames = samples.unfold(0, frame_length, hop_length)
    frame_rms = torch.sqrt(frames.pow(2).mean(dim=1) + 1e-12).numpy()
    if frame_rms.size == 0:
        return {"speech_fraction": 0.0, "snr_proxy_db": 0.0}

    p20 = float(np.percentile(frame_rms, 20))
    p95 = float(np.percentile(frame_rms, 95))
    if p95 < 1e-5:
        return {"speech_fraction": 0.0, "snr_proxy_db": 0.0}

    speech_threshold = max(0.01, p20 * 1.8)
    speech_fraction = float(np.mean(frame_rms > speech_threshold))
    snr_proxy_db = float(20.0 * np.log10((p95 + 1e-8) / (p20 + 1e-8)))
    return {
        "speech_fraction": speech_fraction if np.isfinite(speech_fraction) else 0.0,
        "snr_proxy_db": snr_proxy_db if np.isfinite(snr_proxy_db) else 0.0,
    }


def _open_threshold_file(path, mode):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def load_gate_thresholds(path):
    """Load gate thresholds from JSON or JSON.GZ."""
    with _open_threshold_file(path, "rt") as handle:
        payload = json.load(handle)
    return with_gate_threshold_defaults(payload)


def save_gate_thresholds(thresholds, path):
    """Save gate thresholds to JSON or JSON.GZ."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = with_gate_threshold_defaults(thresholds)
    with _open_threshold_file(path, "wt") as handle:
        json.dump(payload, handle, indent=2)


def apply_gate_signals(
    avg_logprob,
    compression_ratio,
    trigram_rep_count,
    fourgram_rep_count,
    speech_fraction,
    snr_proxy_db,
    num_hyp_words,
    thresholds,
    token_path_distance=0.0,
    beam_logprob_margin=float("inf"),
):
    """Apply reference-free hallucination gate signals for one utterance."""
    thresholds = with_gate_threshold_defaults(thresholds)
    low_logprob = bool(np.isfinite(avg_logprob) and float(avg_logprob) < float(thresholds["T_logprob"]))
    high_compression = bool(
        np.isfinite(compression_ratio)
        and float(compression_ratio) > float(thresholds["T_compression"])
    )
    trigram_repetition = int(trigram_rep_count or 0) >= 1
    fourgram_repetition = int(fourgram_rep_count or 0) >= 1
    ngram_repetition = bool(trigram_repetition or fourgram_repetition)
    low_speech_fraction = bool(
        np.isfinite(speech_fraction)
        and float(speech_fraction) < float(thresholds["T_speech_fraction"])
    )
    low_snr_proxy = bool(
        np.isfinite(snr_proxy_db)
        and float(snr_proxy_db) < float(thresholds["T_snr_proxy_db"])
    )
    has_nontrivial_text = int(num_hyp_words or 0) >= int(thresholds["T_audio_text_min_words"])
    audio_text_mismatch = bool((low_speech_fraction or low_snr_proxy) and has_nontrivial_text)
    acoustic_gate = audio_text_mismatch
    high_token_path_distance = bool(
        np.isfinite(token_path_distance)
        and float(token_path_distance) >= float(thresholds["T_token_path_distance"])
    )
    low_beam_logprob_margin = bool(
        np.isfinite(beam_logprob_margin)
        and float(beam_logprob_margin) <= float(thresholds["T_beam_logprob_margin"])
    )
    unstable_decoding = bool(high_token_path_distance and low_beam_logprob_margin)
    return {
        "avg_logprob_only": low_logprob,
        "compression_ratio_only": high_compression,
        "trigram_repetition_only": trigram_repetition,
        "fourgram_repetition_only": fourgram_repetition,
        "ngram_repetition": ngram_repetition,
        "low_speech_fraction_only": low_speech_fraction,
        "low_snr_proxy_only": low_snr_proxy,
        "audio_text_mismatch": audio_text_mismatch,
        "acoustic_gate": acoustic_gate,
        "token_path_disagreement_only": high_token_path_distance,
        "beam_logprob_margin_only": low_beam_logprob_margin,
        "unstable_decoding": unstable_decoding,
        "combined_gate": bool(low_logprob or high_compression or ngram_repetition or acoustic_gate or unstable_decoding),
    }


def summarize_gate_ablation(flags, hallucination_like, wer, bleu):
    """Summarize gate filtering against evaluation-only labels and metrics."""
    gate_names = [
        "avg_logprob_only",
        "compression_ratio_only",
        "trigram_repetition_only",
        "fourgram_repetition_only",
        "ngram_repetition",
        "low_speech_fraction_only",
        "low_snr_proxy_only",
        "audio_text_mismatch",
        "acoustic_gate",
        "token_path_disagreement_only",
        "beam_logprob_margin_only",
        "unstable_decoding",
        "combined_gate",
    ]
    n_samples = len(hallucination_like)
    hall = [bool(x) for x in hallucination_like]
    wer_vals = [float(x) for x in wer]
    bleu_vals = [float(x) for x in bleu]
    hall_count = sum(hall)
    non_hall_count = n_samples - hall_count

    summaries = {}
    for gate_name in gate_names:
        gate = [bool(row.get(gate_name, False)) for row in flags]
        accepted = [not value for value in gate]
        flagged_count = sum(gate)
        true_positive = sum(gate[idx] and hall[idx] for idx in range(n_samples))
        false_positive = sum(gate[idx] and not hall[idx] for idx in range(n_samples))
        accepted_hall = sum(accepted[idx] and hall[idx] for idx in range(n_samples))
        accepted_count = sum(accepted)
        accepted_wer = [wer_vals[idx] for idx in range(n_samples) if accepted[idx]]
        accepted_bleu = [bleu_vals[idx] for idx in range(n_samples) if accepted[idx]]
        summaries[gate_name] = {
            "n_samples": n_samples,
            "hallucination_like_rate_before_gate": float(hall_count / n_samples) if n_samples else 0.0,
            "gate_flag_rate": float(flagged_count / n_samples) if n_samples else 0.0,
            "accepted_fraction": float(accepted_count / n_samples) if n_samples else 0.0,
            "hallucination_recall": float(true_positive / hall_count) if hall_count else 0.0,
            "gate_precision": float(true_positive / flagged_count) if flagged_count else 0.0,
            "false_positive_rate": float(false_positive / non_hall_count) if non_hall_count else 0.0,
            "hallucination_like_rate_after_gate_among_accepted": (
                float(accepted_hall / accepted_count) if accepted_count else 0.0
            ),
            "WER_before_gate": float(np.mean(wer_vals)) if wer_vals else 0.0,
            "WER_after_gate_among_accepted": float(np.mean(accepted_wer)) if accepted_wer else 0.0,
            "BLEU_before_gate": float(np.mean(bleu_vals)) if bleu_vals else 0.0,
            "BLEU_after_gate_among_accepted": float(np.mean(accepted_bleu)) if accepted_bleu else 0.0,
        }
    return summaries


def load_test_data(tsv_path, clips_dir, max_samples=None):
    """Load test data from TSV."""
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            audio_path = os.path.join(clips_dir, row["path"])
            if os.path.exists(audio_path):
                samples.append({
                    "audio_path": audio_path,
                    "reference": row["sentence"],
                })
    return samples


def apply_perturbation(waveform, sample_rate, perturb_type, amplitude, duration):
    """Apply perturbation to audio waveform."""
    if perturb_type == "onset_noise":
        # Add noise at the beginning of the utterance
        n_samples = int(duration * sample_rate)
        n_samples = min(n_samples, waveform.shape[-1])
        noise = torch.randn_like(waveform[..., :n_samples]) * amplitude
        waveform = waveform.clone()
        waveform[..., :n_samples] = waveform[..., :n_samples] + noise
    elif perturb_type == "full_noise":
        # Add noise throughout the entire utterance
        noise = torch.randn_like(waveform) * amplitude
        waveform = waveform + noise
    elif perturb_type == "reverb":
        # Synthetic room impulse response: dry signal plus decaying delayed taps.
        strength = max(0.0, float(amplitude))
        delay_samples = max(1, int(0.035 * sample_rate))
        tail_seconds = duration if duration > 0 else 0.45
        tail_samples = max(delay_samples + 1, int(tail_seconds * sample_rate))
        impulse = torch.zeros(
            1,
            1,
            tail_samples,
            dtype=waveform.dtype,
            device=waveform.device,
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
        )[..., :waveform.shape[-1]]
        waveform = convolved.reshape(original_shape)
    elif perturb_type == "silence":
        waveform = torch.zeros_like(waveform)
    elif perturb_type == "leading_silence":
        n_samples = max(0, int(duration * sample_rate))
        if n_samples > 0:
            silence = torch.zeros(
                waveform.shape[:-1] + (n_samples,),
                dtype=waveform.dtype,
                device=waveform.device,
            )
            waveform = torch.cat([silence, waveform], dim=-1)
    elif perturb_type == "speech_band_noise":
        noise = torch.randn_like(waveform)
        spectrum = torch.fft.rfft(noise, dim=-1)
        freqs = torch.fft.rfftfreq(noise.shape[-1], d=1.0 / sample_rate).to(noise.device)
        mask = ((freqs >= 300.0) & (freqs <= 3400.0)).to(spectrum.dtype)
        filtered = torch.fft.irfft(spectrum * mask, n=noise.shape[-1], dim=-1)
        filtered = filtered / (filtered.pow(2).mean().sqrt() + 1e-8)
        signal_rms = waveform.pow(2).mean().sqrt().clamp_min(1e-4)
        waveform = waveform + filtered * signal_rms * float(amplitude)
    elif perturb_type == "none":
        pass
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    # Clip to valid range
    waveform = torch.clamp(waveform, -1.0, 1.0)
    return waveform


def transcribe_batch(model, processor, audio_paths, perturb_type="none",
                     perturb_amplitude=0.0, perturb_duration=0.0,
                     device="cuda", batch_size=16, return_decoder_signals=False,
                     return_audio_signals=False, num_decoder_hypotheses=1,
                     decoder_hypothesis_beams=None, return_alternates=False,
                     repetition_penalty=1.0):
    """Transcribe audio files, optionally with perturbation."""
    hypotheses = []
    avg_logprobs = []
    audio_signals = []
    alternate_summaries = []
    num_decoder_hypotheses = max(1, int(num_decoder_hypotheses or 1))
    decoder_hypothesis_beams = max(
        num_decoder_hypotheses,
        int(decoder_hypothesis_beams or num_decoder_hypotheses),
    )

    for batch_start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[batch_start:batch_start + batch_size]
        waveforms = []

        for path in batch_paths:
            waveform, sample_rate = torchaudio.load(path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Apply perturbation
            waveform = apply_perturbation(
                waveform, 16000, perturb_type, perturb_amplitude, perturb_duration
            )

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if return_audio_signals:
                audio_signals.append(compute_audio_gate_features(waveform, sample_rate=16000))

            waveforms.append(waveform.squeeze().numpy())

        # Batch feature extraction (significantly faster than per-file)
        input_features = processor.feature_extractor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = input_features.to(device)
        model_dtype = next(model.parameters()).dtype
        input_features["input_features"] = input_features["input_features"].to(dtype=model_dtype)

        with torch.no_grad():
            if return_decoder_signals:
                generate_kwargs = {
                    "attention_mask": input_features.get("attention_mask", None),
                    "max_new_tokens": 225,
                    "language": "en",
                    "task": "transcribe",
                    "repetition_penalty": repetition_penalty,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if num_decoder_hypotheses > 1:
                    generate_kwargs.update({
                        "num_beams": decoder_hypothesis_beams,
                        "num_return_sequences": num_decoder_hypotheses,
                    })
                gen_out = model.generate(
                    input_features["input_features"],
                    **generate_kwargs,
                )
                predicted_ids = gen_out.sequences
                generated_logprobs = compute_avg_logprobs_from_generate(gen_out, model=model)
                if num_decoder_hypotheses > 1:
                    primary_ids = predicted_ids[::num_decoder_hypotheses]
                    primary_transcriptions = processor.tokenizer.batch_decode(
                        primary_ids, skip_special_tokens=True
                    )
                    special_token_ids = set(processor.tokenizer.all_special_ids)
                    for group_idx, offset in enumerate(range(0, int(predicted_ids.shape[0]), num_decoder_hypotheses)):
                        sequence_group = predicted_ids[offset:offset + num_decoder_hypotheses].detach().cpu().tolist()
                        token_path_group = [
                            compact_token_path(token_ids, special_token_ids)
                            for token_ids in sequence_group
                        ]
                        logprob_group = generated_logprobs[offset:offset + num_decoder_hypotheses]
                        hypotheses.append(primary_transcriptions[group_idx] if group_idx < len(primary_transcriptions) else "")
                        avg_logprobs.append(logprob_group[0] if logprob_group else 0.0)
                        alternate_summaries.append(
                            summarize_decoder_token_paths(token_path_group, logprob_group)
                        )
                    predicted_ids = None
                else:
                    avg_logprobs.extend(generated_logprobs)
            else:
                predicted_ids = model.generate(
                    input_features["input_features"],
                    attention_mask=input_features.get("attention_mask", None),
                    max_new_tokens=225,
                    language="en",
                    task="transcribe",
                    repetition_penalty=repetition_penalty,
                )

        if predicted_ids is not None:
            transcriptions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            hypotheses.extend(transcriptions)

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Transcribed {min(batch_start + batch_size, len(audio_paths))}/{len(audio_paths)}")

    if return_decoder_signals:
        if return_alternates:
            if return_audio_signals:
                return hypotheses, avg_logprobs, audio_signals, alternate_summaries
            return hypotheses, avg_logprobs, alternate_summaries
        if return_audio_signals:
            return hypotheses, avg_logprobs, audio_signals
        return hypotheses, avg_logprobs
    if return_audio_signals:
        return hypotheses, audio_signals
    return hypotheses


def compute_bleu_scores(hypotheses, references):
    """Compute sentence-level BLEU-4 with smoothing via sacrebleu."""
    import sacrebleu

    results = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip():
            results.append(0.0)
            continue
        bleu_val = sacrebleu.sentence_bleu(
            hyp, [ref], smooth_method="exp"
        ).score
        # sacrebleu returns 0-100 scale; normalize to 0-1
        results.append(bleu_val / 100.0)

    return results


def compute_repetitions(text, min_repeats=2):
    """Count n-gram repetitions in text."""
    tokens = text.lower().split()
    results = {}

    for n in [2, 3, 4]:
        if len(tokens) < n:
            results[f"{n}gram_repeats"] = 0
            continue
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for count in counts.values() if count >= min_repeats)
        results[f"{n}gram_repeats"] = repeated

    return results


def word_error_rate_between_texts(reference_text, hypothesis_text):
    """Reference-free text distance: WER between two decoded hypotheses."""
    ref_tokens = str(reference_text).split()
    hyp_tokens = str(hypothesis_text).split()
    if not ref_tokens and not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 1.0

    prev = list(range(len(hyp_tokens) + 1))
    for ref_idx, ref_token in enumerate(ref_tokens, start=1):
        curr = [ref_idx]
        for hyp_idx, hyp_token in enumerate(hyp_tokens, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            curr.append(min(
                curr[hyp_idx - 1] + 1,
                prev[hyp_idx] + 1,
                prev[hyp_idx - 1] + substitution_cost,
            ))
        prev = curr
    return prev[-1] / len(ref_tokens)


def token_edit_distance(left_tokens, right_tokens):
    """Normalized edit distance between two token-id paths."""
    left = [int(token) for token in left_tokens]
    right = [int(token) for token in right_tokens]
    denominator = max(len(left), len(right), 1)
    prev = list(range(len(right) + 1))
    for left_idx, left_token in enumerate(left, start=1):
        curr = [left_idx]
        for right_idx, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            curr.append(min(
                curr[right_idx - 1] + 1,
                prev[right_idx] + 1,
                prev[right_idx - 1] + substitution_cost,
            ))
        prev = curr
    return prev[-1] / denominator


def first_token_divergence(left_tokens, right_tokens):
    """Return first differing token position, or -1 for identical paths."""
    for idx, (left_token, right_token) in enumerate(zip(left_tokens, right_tokens)):
        if int(left_token) != int(right_token):
            return idx
    if len(left_tokens) != len(right_tokens):
        return min(len(left_tokens), len(right_tokens))
    return -1


def compact_token_path(token_ids, special_token_ids):
    """Remove prompt/control tokens before compact token-path comparison."""
    special_ids = set(int(token) for token in special_token_ids)
    return [int(token) for token in token_ids if int(token) not in special_ids]


def token_path_hash(token_ids):
    payload = ",".join(str(int(token)) for token in token_ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def summarize_decoder_token_paths(token_paths, path_scores):
    """Summarize beam alternatives using token-id paths, not decoded text."""
    normalized_paths = [[int(token) for token in path] for path in token_paths]
    unique_paths = list(dict.fromkeys(tuple(path) for path in normalized_paths))
    pairwise_distances = []
    divergence_steps = []
    for left_idx in range(len(unique_paths)):
        for right_idx in range(left_idx + 1, len(unique_paths)):
            left_path = unique_paths[left_idx]
            right_path = unique_paths[right_idx]
            pairwise_distances.append(token_edit_distance(left_path, right_path))
            divergence = first_token_divergence(left_path, right_path)
            if divergence >= 0:
                divergence_steps.append(divergence)

    finite_scores = [float(value) for value in path_scores if np.isfinite(value)]
    if len(unique_paths) >= 2 and len(finite_scores) >= 2:
        sorted_logprobs = sorted(finite_scores, reverse=True)
        beam_logprob_margin = sorted_logprobs[0] - sorted_logprobs[1]
        beam_logprob_spread = sorted_logprobs[0] - sorted_logprobs[-1]
    else:
        beam_logprob_margin = float("inf")
        beam_logprob_spread = 0.0

    token_lengths = [len(path) for path in normalized_paths]
    token_length_spread = max(token_lengths) - min(token_lengths) if token_lengths else 0
    return {
        "num_token_paths": len(normalized_paths),
        "unique_token_paths": len(unique_paths),
        "mean_token_path_distance": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
        "max_token_path_distance": float(np.max(pairwise_distances)) if pairwise_distances else 0.0,
        "first_divergence_step": int(min(divergence_steps)) if divergence_steps else -1,
        "token_length_spread": int(token_length_spread),
        "beam_logprob_margin": float(beam_logprob_margin),
        "beam_logprob_spread": float(beam_logprob_spread),
        "token_path_hashes": [token_path_hash(path) for path in normalized_paths],
        "token_lengths": token_lengths,
        "path_scores": path_scores,
    }


def summarize_decoder_hypotheses(hypotheses, avg_logprobs):
    """Backward-compatible text disagreement summary for old callers."""
    norm_hypotheses = [normalizer(hyp).strip() for hyp in hypotheses]
    pairwise_distances = []
    for left_idx in range(len(norm_hypotheses)):
        for right_idx in range(left_idx + 1, len(norm_hypotheses)):
            pairwise_distances.append(
                word_error_rate_between_texts(norm_hypotheses[left_idx], norm_hypotheses[right_idx])
            )
    finite_logprobs = [float(value) for value in avg_logprobs if np.isfinite(value)]
    if len(finite_logprobs) >= 2:
        sorted_logprobs = sorted(finite_logprobs, reverse=True)
        logprob_margin = sorted_logprobs[0] - sorted_logprobs[1]
        logprob_spread = sorted_logprobs[0] - sorted_logprobs[-1]
    else:
        logprob_margin = float("inf")
        logprob_spread = 0.0
    return {
        "hypotheses": hypotheses,
        "avg_logprobs": avg_logprobs,
        "mean_pairwise_hyp_wer": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
        "max_pairwise_hyp_wer": float(np.max(pairwise_distances)) if pairwise_distances else 0.0,
        "logprob_margin": float(logprob_margin),
        "logprob_spread": float(logprob_spread),
    }


def compute_lm_perplexity(texts, model_name="Qwen/Qwen3-1.7B", device="cuda", batch_size=8):
    """
    Compute sequence plausibility using a causal LM.
    Returns normalized log-probabilities per sentence.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading LM ({model_name}) for plausibility scoring...")

    _dtype = torch.float16 if "cuda" in device else torch.float32
    load_kwargs = {"dtype": _dtype}

    lm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, **load_kwargs
    ).to(device)
    lm_model.eval()

    # Set pad token if not set
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    scores = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        # Filter empty texts
        batch_scores = []
        for text in batch_texts:
            text = text.strip()
            if not text:
                batch_scores.append(0.0)
                continue

            encodings = lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            input_ids = encodings.input_ids

            with torch.no_grad():
                outputs = lm_model(input_ids, labels=input_ids)
                # Negative log-likelihood per token (lower = more plausible)
                nll = outputs.loss.item()

            # Convert to probability-like score (higher = more plausible)
            # Using exp(-nll) gives the geometric mean token probability
            prob_score = np.exp(-nll) if np.isfinite(nll) else 0.0
            if not np.isfinite(prob_score):
                prob_score = 0.0
            batch_scores.append(prob_score)

        scores.extend(batch_scores)

        if (batch_start // batch_size) % 20 == 0:
            print(f"  Scored {min(batch_start + batch_size, len(texts))}/{len(texts)}")

    lm_model.cpu()
    del lm_model
    torch.cuda.empty_cache()

    return scores


def finite_float(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def _shorten_for_log(text, max_chars=220):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def print_qualitative_examples(
    references,
    hypotheses,
    per_sample_wer,
    per_sample_wacc,
    norm_plausibility,
    hallucination_like,
    num_examples=3,
):
    if not hypotheses or num_examples <= 0:
        return

    mean_wer = float(np.mean(per_sample_wer)) if per_sample_wer else 0.0
    mean_plausibility = float(np.mean(norm_plausibility)) if norm_plausibility else 0.0

    rows = []
    for idx, (ref, hyp, wer, wacc, plaus, hall) in enumerate(
        zip(
            references,
            hypotheses,
            per_sample_wer,
            per_sample_wacc,
            norm_plausibility,
            hallucination_like,
        )
    ):
        reps = compute_repetitions(hyp)
        rep_total = reps["2gram_repeats"] + reps["3gram_repeats"] + reps["4gram_repeats"]
        rows.append({
            "idx": idx,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer,
            "wacc": wacc,
            "plausibility": plaus,
            "rep_total": rep_total,
            "hall_like": bool(hall),
        })

    healthy = sorted(
        rows,
        key=lambda r: (r["wer"], r["rep_total"], -r["plausibility"], r["idx"]),
    )[:num_examples]

    unhealthy_pool = [
        r for r in rows
        if r["hall_like"] or r["rep_total"] >= 2
    ]
    if not unhealthy_pool:
        unhealthy_pool = rows
    unhealthy = sorted(
        unhealthy_pool,
        key=lambda r: (
            not r["hall_like"],
            -r["rep_total"],
            -r["wer"],
            -r["plausibility"],
            r["idx"],
        ),
    )[:num_examples]

    print("\nQualitative examples:")
    print(f"  Thresholds: mean WER={mean_wer:.4f}, mean plausibility={mean_plausibility:.4f}")
    for title, examples in [("Healthy", healthy), ("Unhealthy", unhealthy)]:
        print(f"  {title} examples:")
        for rank, row in enumerate(examples, start=1):
            print(
                f"    {rank}. idx={row['idx']} WAcc={row['wacc']:.4f} "
                f"WER={row['wer']:.4f} plaus={row['plausibility']:.4f} "
                f"reps={row['rep_total']} hall_like={int(row['hall_like'])}"
            )
            print(f"       REF: {_shorten_for_log(row['reference'])}")
            print(f"       HYP: {_shorten_for_log(row['hypothesis'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model (LoRA checkpoint)")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3",
                        help="Base Whisper model name")
    parser.add_argument("--test_tsv", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True,
                        help="Name of noise config (base/uu/rr/ru/ur)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for debugging)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lm_model", type=str, default="Qwen/Qwen3-1.7B",
                        help="Language model for plausibility scoring")
    parser.add_argument("--lm_models", type=str, nargs="*", default=None,
                        help="Multiple language models for plausibility scoring")
    parser.add_argument("--eval_mode", choices=["normal", "gated"], default="normal",
                        help="normal keeps historical outputs; gated adds decoder-only gate outputs")
    parser.add_argument("--gate_thresholds_path", type=str, default=None,
                        help="Gated mode only: load gate thresholds from JSON or JSON.GZ")
    parser.add_argument("--calibrate_gate", action="store_true",
                        help="Gated mode only: calibrate gate thresholds from this eval run")
    parser.add_argument("--save_gate_thresholds_path", type=str, default=None,
                        help="Gated mode only: save calibrated gate thresholds to JSON or JSON.GZ")
    parser.add_argument("--gate_speech_fraction_threshold", type=float, default=None,
                        help="Gated mode only: flag low speech if speech_fraction is below this value")
    parser.add_argument("--gate_snr_proxy_threshold", type=float, default=None,
                        help="Gated mode only: flag noisy/flat audio if snr_proxy_db is below this value")
    parser.add_argument("--gate_audio_text_min_words", type=int, default=None,
                        help="Gated mode only: minimum decoded words for acoustic/text mismatch")
    parser.add_argument("--num_decoder_hypotheses", type=int, default=1,
                        help="Gated mode only: return this many beam token paths for instability gating")
    parser.add_argument("--decoder_hypothesis_beams", type=int, default=None,
                        help="Gated mode only: beam size for token-path alternatives")
    parser.add_argument("--gate_token_path_distance_threshold", type=float, default=None,
                        help="Gated mode only: mean pairwise token-path distance threshold for instability")
    parser.add_argument("--gate_beam_logprob_margin_threshold", type=float, default=None,
                        help="Gated mode only: max top-2 beam score margin for unstable decoding")
    parser.add_argument("--gate_hypothesis_disagreement_threshold", type=float, default=None,
                        help="Deprecated alias for --gate_token_path_distance_threshold")
    parser.add_argument("--gate_logprob_margin_threshold", type=float, default=None,
                        help="Deprecated alias for --gate_beam_logprob_margin_threshold")

    # Perturbation args
    parser.add_argument("--perturb_type", type=str, default="none",
                        choices=[
                            "none", "onset_noise", "full_noise", "reverb",
                            "silence", "leading_silence", "speech_band_noise",
                        ])
    parser.add_argument("--perturb_amplitude", type=float, default=0.0)
    parser.add_argument("--perturb_duration", type=float, default=0.0,
                        help="Duration of onset noise in seconds")
    parser.add_argument("--num_log_examples", type=int, default=3,
                        help="Number of healthy/unhealthy examples to print per run")

    args = parser.parse_args()
    gated_mode = args.eval_mode == "gated"
    if not gated_mode and (args.gate_thresholds_path or args.calibrate_gate or args.save_gate_thresholds_path):
        parser.error("Gate threshold options require --eval_mode gated")
    if not gated_mode and args.num_decoder_hypotheses != 1:
        parser.error("--num_decoder_hypotheses requires --eval_mode gated")
    if args.num_decoder_hypotheses < 1:
        parser.error("--num_decoder_hypotheses must be at least 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    perturb_tag = args.perturb_type
    if args.perturb_type != "none":
        perturb_tag = f"{args.perturb_type}_amp{args.perturb_amplitude}_dur{args.perturb_duration}"

    print(f"=== Evaluation: {args.config_name} | Perturbation: {perturb_tag} ===")
    print(f"Eval mode: {args.eval_mode}")

    # Load model
    print("Loading model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    adapter_config = os.path.join(args.model_dir, "adapter_config.json")
    adapter_safetensors = os.path.join(args.model_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(args.model_dir, "adapter_model.bin")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(
            f"Missing adapter_config.json in --model_dir: {args.model_dir}. "
            "Pass a PEFT checkpoint directory, e.g. base/checkpoint-10000, "
            "rr_64pct/checkpoint-9375, or ru_64pct/checkpoint-9375."
        )
    if not (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)):
        raise FileNotFoundError(
            f"Missing adapter weights in --model_dir: {args.model_dir}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model = model.to(device)
    model.eval()

    # Clear forced_decoder_ids set by processor to avoid conflict with task=transcribe
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    processor = WhisperProcessor.from_pretrained(args.base_model, language="en", task="transcribe")

    # Load test data
    print("Loading test data...")
    samples = load_test_data(args.test_tsv, args.clips_dir, args.max_samples)
    print(f"  Test samples: {len(samples)}")

    audio_paths = [s["audio_path"] for s in samples]
    references = [s["reference"] for s in samples]

    # Transcribe
    print("Transcribing...")
    decoder_hypothesis_info = None
    if gated_mode:
        transcribe_result = transcribe_batch(
            model, processor, audio_paths,
            perturb_type=args.perturb_type,
            perturb_amplitude=args.perturb_amplitude,
            perturb_duration=args.perturb_duration,
            device=device,
            batch_size=args.batch_size,
            return_decoder_signals=True,
            return_audio_signals=True,
            num_decoder_hypotheses=args.num_decoder_hypotheses,
            decoder_hypothesis_beams=args.decoder_hypothesis_beams,
            return_alternates=args.num_decoder_hypotheses > 1,
        )
        if args.num_decoder_hypotheses > 1:
            hypotheses, avg_logprobs, audio_signals, decoder_hypothesis_info = transcribe_result
        else:
            hypotheses, avg_logprobs, audio_signals = transcribe_result
    else:
        hypotheses = transcribe_batch(
            model, processor, audio_paths,
            perturb_type=args.perturb_type,
            perturb_amplitude=args.perturb_amplitude,
            perturb_duration=args.perturb_duration,
            device=device,
            batch_size=args.batch_size,
            return_decoder_signals=False,
        )
        avg_logprobs = None
        audio_signals = None

    # Free Whisper model memory
    model.cpu()
    del model, base_model
    torch.cuda.empty_cache()

    # Normalize
    norm_hyps = [normalizer(h).strip() for h in hypotheses]
    norm_refs = [normalizer(r).strip() for r in references]

    # --- Metric 1: Lexical Accuracy (WAcc) ---
    metric_wer = evaluate.load("wer")
    # Filter empty references
    valid_pairs = [(h, r) for h, r in zip(norm_hyps, norm_refs) if len(r) > 0]
    valid_hyps, valid_refs = zip(*valid_pairs) if valid_pairs else ([], [])

    wer = metric_wer.compute(predictions=list(valid_hyps), references=list(valid_refs))
    wacc = 1.0 - wer
    print(f"\n  WER: {wer:.4f}")
    print(f"  WAcc: {wacc:.4f}")

    # Per-sample WER/WAcc for joint analysis. WAcc follows the paper exactly:
    # WAcc = 1 - WER, without clipping negative values for insertion-heavy cases.
    import jiwer
    per_sample_wer = [
        jiwer.wer(r, h) if len(r) > 0 else 1.0
        for h, r in zip(norm_hyps, norm_refs)
    ]
    avg_sample_wer = float(np.mean(per_sample_wer)) if per_sample_wer else 0.0
    per_sample_wacc = [1.0 - w for w in per_sample_wer]
    avg_sample_wacc = float(np.mean(per_sample_wacc)) if per_sample_wacc else 0.0

    # --- Metric 2: Sequence Plausibility (LM scoring) ---
    # Determine LM models to use
    lm_models = args.lm_models if args.lm_models else [args.lm_model]

    all_lm_results = {}
    for lm_name in lm_models:
        lm_short = lm_name.split("/")[-1]
        print(f"\nComputing sequence plausibility with {lm_short}...")
        hyp_plausibility = compute_lm_perplexity(norm_hyps, model_name=lm_name, device=device)
        ref_plausibility = compute_lm_perplexity(norm_refs, model_name=lm_name, device=device)

        # Normalized plausibility (hypothesis / reference), clipped to [0, 1]
        norm_plausibility = []
        for hp, rp in zip(hyp_plausibility, ref_plausibility):
            hp = finite_float(hp)
            rp = finite_float(rp)
            if rp > 0:
                norm_plausibility.append(finite_float(min(hp / rp, 1.0)))
            else:
                norm_plausibility.append(0.0)

        avg_plausibility = finite_float(np.mean(norm_plausibility))
        avg_raw_plausibility = finite_float(np.mean(hyp_plausibility))
        print(f"  {lm_short} avg normalized plausibility: {avg_plausibility:.4f}")
        print(f"  {lm_short} avg raw plausibility: {avg_raw_plausibility:.4f}")

        all_lm_results[lm_short] = {
            "hyp_plausibility": hyp_plausibility,
            "ref_plausibility": ref_plausibility,
            "norm_plausibility": norm_plausibility,
            "avg_plausibility": avg_plausibility,
            "avg_raw_plausibility": avg_raw_plausibility,
        }

    # Use first LM's results for per-sample output
    first_lm = list(all_lm_results.keys())[0]
    hyp_plausibility = all_lm_results[first_lm]["hyp_plausibility"]
    norm_plausibility = all_lm_results[first_lm]["norm_plausibility"]
    avg_plausibility = all_lm_results[first_lm]["avg_plausibility"]
    avg_raw_plausibility = all_lm_results[first_lm]["avg_raw_plausibility"]

    # Canonical hallucination-like criterion: above-average sentence probability
    # and above-average WER within this evaluation set.
    hallucination_like = [
        bool((np_score > avg_plausibility) and (sample_wer > avg_sample_wer))
        for np_score, sample_wer in zip(norm_plausibility, per_sample_wer)
    ]
    hallucination_like_count = int(sum(hallucination_like))
    hallucination_like_rate = (
        float(hallucination_like_count / len(hallucination_like))
        if hallucination_like
        else 0.0
    )
    print(
        "  Hallucination-like: "
        f"{hallucination_like_count}/{len(hallucination_like)} "
        f"({hallucination_like_rate:.4f}); "
        f"thresholds: norm_plausibility>{avg_plausibility:.4f}, "
        f"wer>{avg_sample_wer:.4f}"
    )

    # --- Metric 3: BLEU ---
    print("\nComputing BLEU scores...")
    bleu_scores = compute_bleu_scores(norm_hyps, norm_refs)
    mean_bleu = np.mean(bleu_scores)
    print(f"  Mean BLEU: {mean_bleu:.4f}")

    # --- Metric 4: Repetition Analysis ---
    print("\nAnalyzing repetitions...")
    total_rep_2gram = 0
    total_rep_3gram = 0
    total_rep_4gram = 0
    sentences_with_reps_2 = 0
    sentences_with_reps_3 = 0
    sentences_with_reps_4 = 0

    for h in norm_hyps:
        reps = compute_repetitions(h)
        if reps["2gram_repeats"] > 0:
            sentences_with_reps_2 += 1
            total_rep_2gram += reps["2gram_repeats"]
        if reps["3gram_repeats"] > 0:
            sentences_with_reps_3 += 1
            total_rep_3gram += reps["3gram_repeats"]
        if reps["4gram_repeats"] > 0:
            sentences_with_reps_4 += 1
            total_rep_4gram += reps["4gram_repeats"]

    print(f"  Sentences with bigram repeats: {sentences_with_reps_2}")
    print(f"  Sentences with trigram repeats: {sentences_with_reps_3}")
    print(f"  Sentences with 4-gram repeats: {sentences_with_reps_4}")

    gate_thresholds = None
    gate_threshold_source = None
    compression_ratios = None
    gate_flags = None
    gate_ablation = None
    if gated_mode:
        print("\nComputing decoder-only gate signals...")
        compression_ratios = [compute_compression_ratio(text) for text in norm_hyps]
        if args.gate_thresholds_path:
            gate_thresholds = load_gate_thresholds(args.gate_thresholds_path)
            gate_threshold_source = args.gate_thresholds_path
        else:
            gate_thresholds = calibrate_gate_thresholds(avg_logprobs, compression_ratios)
            gate_threshold_source = "current_run_calibration"
            if not args.calibrate_gate:
                print("  No gate thresholds supplied; self-calibrating from this run.")
        if args.gate_speech_fraction_threshold is not None:
            gate_thresholds["T_speech_fraction"] = float(args.gate_speech_fraction_threshold)
        if args.gate_snr_proxy_threshold is not None:
            gate_thresholds["T_snr_proxy_db"] = float(args.gate_snr_proxy_threshold)
        if args.gate_audio_text_min_words is not None:
            gate_thresholds["T_audio_text_min_words"] = int(args.gate_audio_text_min_words)
        if args.gate_token_path_distance_threshold is not None:
            gate_thresholds["T_token_path_distance"] = float(args.gate_token_path_distance_threshold)
        elif args.gate_hypothesis_disagreement_threshold is not None:
            gate_thresholds["T_token_path_distance"] = float(args.gate_hypothesis_disagreement_threshold)
        if args.gate_beam_logprob_margin_threshold is not None:
            gate_thresholds["T_beam_logprob_margin"] = float(args.gate_beam_logprob_margin_threshold)
        elif args.gate_logprob_margin_threshold is not None:
            gate_thresholds["T_beam_logprob_margin"] = float(args.gate_logprob_margin_threshold)
        gate_thresholds = with_gate_threshold_defaults(gate_thresholds)
        if args.save_gate_thresholds_path:
            save_gate_thresholds(gate_thresholds, args.save_gate_thresholds_path)
            print(f"  Saved gate thresholds: {args.save_gate_thresholds_path}")

        gate_flags = []
        for idx, hyp in enumerate(norm_hyps):
            reps = compute_repetitions(hyp)
            token_path_distance = 0.0
            beam_logprob_margin = float("inf")
            if decoder_hypothesis_info:
                token_path_distance = decoder_hypothesis_info[idx]["mean_token_path_distance"]
                beam_logprob_margin = decoder_hypothesis_info[idx]["beam_logprob_margin"]
            gate_flags.append(
                apply_gate_signals(
                    avg_logprobs[idx],
                    compression_ratios[idx],
                    reps["3gram_repeats"],
                    reps["4gram_repeats"],
                    audio_signals[idx]["speech_fraction"],
                    audio_signals[idx]["snr_proxy_db"],
                    len(norm_hyps[idx].split()),
                    gate_thresholds,
                    token_path_distance=token_path_distance,
                    beam_logprob_margin=beam_logprob_margin,
                )
            )
        gate_ablation = summarize_gate_ablation(
            gate_flags,
            hallucination_like,
            per_sample_wer,
            bleu_scores,
        )
        print(
            "  Gate thresholds: "
            f"T_logprob={gate_thresholds['T_logprob']:.4f}, "
            f"T_compression={gate_thresholds['T_compression']:.4f}, "
            f"T_token_path_distance={gate_thresholds['T_token_path_distance']:.4f}, "
            f"T_beam_logprob_margin={gate_thresholds['T_beam_logprob_margin']:.4f}"
        )
        print(
            f"  Gate flagged: {sum(row['combined_gate'] for row in gate_flags)}/{len(gate_flags)}"
        )

    print_qualitative_examples(
        norm_refs,
        norm_hyps,
        per_sample_wer,
        per_sample_wacc,
        norm_plausibility,
        hallucination_like,
        num_examples=args.num_log_examples,
    )

    # --- Summary ---
    results = {
        "config": args.config_name,
        "perturbation": perturb_tag,
        "n_samples": int(len(samples)),
        "wer": round(finite_float(wer), 4),
        "wacc": round(finite_float(wacc), 4),
        "mean_sample_wacc": round(finite_float(avg_sample_wacc), 4),
        "avg_normalized_plausibility": round(finite_float(avg_plausibility), 4),
        "avg_raw_plausibility": round(finite_float(avg_raw_plausibility), 4),
        "hallucination_like_count": hallucination_like_count,
        "hallucination_like_rate": round(finite_float(hallucination_like_rate), 4),
        "hallucination_wer_threshold": round(finite_float(avg_sample_wer), 4),
        "hallucination_plausibility_threshold": round(finite_float(avg_plausibility), 4),
        "mean_bleu": round(finite_float(mean_bleu), 4),
        "sentences_with_bigram_repeats": int(sentences_with_reps_2),
        "sentences_with_trigram_repeats": int(sentences_with_reps_3),
        "sentences_with_4gram_repeats": int(sentences_with_reps_4),
    }
    if gated_mode:
        results.update({
            "eval_mode": args.eval_mode,
            "mean_avg_logprob": round(finite_float(np.mean(avg_logprobs)), 4) if avg_logprobs else 0.0,
            "mean_compression_ratio": round(finite_float(np.mean(compression_ratios)), 4) if compression_ratios else 0.0,
            "mean_speech_fraction": round(finite_float(np.mean([row["speech_fraction"] for row in audio_signals])), 4) if audio_signals else 0.0,
            "mean_snr_proxy_db": round(finite_float(np.mean([row["snr_proxy_db"] for row in audio_signals])), 4) if audio_signals else 0.0,
            "num_decoder_hypotheses": int(args.num_decoder_hypotheses),
            "mean_token_path_distance": round(finite_float(np.mean([
                row["mean_token_path_distance"] for row in decoder_hypothesis_info
            ])), 4) if decoder_hypothesis_info else 0.0,
            "mean_max_token_path_distance": round(finite_float(np.mean([
                row["max_token_path_distance"] for row in decoder_hypothesis_info
            ])), 4) if decoder_hypothesis_info else 0.0,
            "mean_unique_token_paths": round(finite_float(np.mean([
                row["unique_token_paths"] for row in decoder_hypothesis_info
            ])), 4) if decoder_hypothesis_info else 0.0,
            "mean_beam_logprob_margin": round(finite_float(np.mean([
                row["beam_logprob_margin"] for row in decoder_hypothesis_info
            ])), 4) if decoder_hypothesis_info else 0.0,
            "mean_beam_logprob_spread": round(finite_float(np.mean([
                row["beam_logprob_spread"] for row in decoder_hypothesis_info
            ])), 4) if decoder_hypothesis_info else 0.0,
            "gate_thresholds": gate_thresholds,
            "gate_threshold_source": gate_threshold_source,
            "gate_flag_rate": round(
                finite_float(np.mean([row["combined_gate"] for row in gate_flags])), 4
            ) if gate_flags else 0.0,
            "gate_ablation": gate_ablation,
        })

    # Save results
    mode_suffix = "_gated" if gated_mode else ""
    results_path = os.path.join(args.output_dir, f"results_{args.config_name}_{perturb_tag}{mode_suffix}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {results_path}")

    # Save per-sample details for analysis
    details_path = os.path.join(args.output_dir, f"details_{args.config_name}_{perturb_tag}{mode_suffix}.tsv")
    with open(details_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        detail_header = [
            "reference", "hypothesis", "wer", "wacc", "plausibility",
            "norm_plausibility", "hallucination_like",
            "hallucination_wer_threshold", "hallucination_plausibility_threshold",
            "2gram_reps", "3gram_reps", "4gram_reps",
        ]
        if gated_mode:
            detail_header.extend([
                "avg_logprob", "compression_ratio", "speech_fraction", "snr_proxy_db",
                "gate_avg_logprob_only", "gate_compression_ratio_only",
                "gate_trigram_repetition_only", "gate_fourgram_repetition_only",
                "gate_ngram_repetition", "gate_low_speech_fraction_only",
                "gate_low_snr_proxy_only", "gate_audio_text_mismatch",
                "gate_acoustic", "decoder_num_token_paths",
                "decoder_unique_token_paths", "decoder_mean_token_path_distance",
                "decoder_max_token_path_distance", "decoder_first_divergence_step",
                "decoder_token_length_spread", "decoder_beam_logprob_margin",
                "decoder_beam_logprob_spread", "decoder_token_path_hashes",
                "decoder_token_lengths", "decoder_path_scores",
                "gate_token_path_disagreement_only",
                "gate_beam_logprob_margin_only", "gate_unstable_decoding",
                "gate_flagged",
            ])
        writer.writerow(detail_header)
        for i in range(len(norm_hyps)):
            reps = compute_repetitions(norm_hyps[i])
            row = [
                norm_refs[i], norm_hyps[i],
                f"{per_sample_wer[i]:.4f}",
                f"{per_sample_wacc[i]:.4f}",
                f"{hyp_plausibility[i]:.4f}",
                f"{norm_plausibility[i]:.4f}",
                int(hallucination_like[i]),
                f"{avg_sample_wer:.4f}",
                f"{avg_plausibility:.4f}",
                reps["2gram_repeats"], reps["3gram_repeats"], reps["4gram_repeats"],
            ]
            if gated_mode:
                hypothesis_summary = decoder_hypothesis_info[i] if decoder_hypothesis_info else None
                row.extend([
                    f"{avg_logprobs[i]:.4f}",
                    f"{compression_ratios[i]:.4f}",
                    f"{audio_signals[i]['speech_fraction']:.4f}",
                    f"{audio_signals[i]['snr_proxy_db']:.4f}",
                    int(gate_flags[i]["avg_logprob_only"]),
                    int(gate_flags[i]["compression_ratio_only"]),
                    int(gate_flags[i]["trigram_repetition_only"]),
                    int(gate_flags[i]["fourgram_repetition_only"]),
                    int(gate_flags[i]["ngram_repetition"]),
                    int(gate_flags[i]["low_speech_fraction_only"]),
                    int(gate_flags[i]["low_snr_proxy_only"]),
                    int(gate_flags[i]["audio_text_mismatch"]),
                    int(gate_flags[i]["acoustic_gate"]),
                    hypothesis_summary["num_token_paths"] if hypothesis_summary else "",
                    hypothesis_summary["unique_token_paths"] if hypothesis_summary else "",
                    f"{hypothesis_summary['mean_token_path_distance']:.4f}" if hypothesis_summary else "",
                    f"{hypothesis_summary['max_token_path_distance']:.4f}" if hypothesis_summary else "",
                    hypothesis_summary["first_divergence_step"] if hypothesis_summary else "",
                    hypothesis_summary["token_length_spread"] if hypothesis_summary else "",
                    f"{hypothesis_summary['beam_logprob_margin']:.4f}" if hypothesis_summary else "",
                    f"{hypothesis_summary['beam_logprob_spread']:.4f}" if hypothesis_summary else "",
                    json.dumps(hypothesis_summary["token_path_hashes"]) if hypothesis_summary else "",
                    json.dumps(hypothesis_summary["token_lengths"]) if hypothesis_summary else "",
                    json.dumps([
                        round(finite_float(value), 4)
                        for value in hypothesis_summary["path_scores"]
                    ]) if hypothesis_summary else "",
                    int(gate_flags[i]["token_path_disagreement_only"]),
                    int(gate_flags[i]["beam_logprob_margin_only"]),
                    int(gate_flags[i]["unstable_decoding"]),
                    int(gate_flags[i]["combined_gate"]),
                ])
            writer.writerow(row)
    print(f"Per-sample details saved to {details_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Config: {args.config_name} | Perturbation: {perturb_tag}")
    print(f"  WAcc:          {wacc:.4f}")
    print(f"  Plausibility:  {avg_plausibility:.4f}")
    print(f"  Hall-like:     {hallucination_like_rate:.4f}")
    print(f"  Bigram reps:   {sentences_with_reps_2}")
    print(f"  Trigram reps:  {sentences_with_reps_3}")
    print(f"  4-gram reps:   {sentences_with_reps_4}")
    if gated_mode:
        print(f"  Gate flagged:  {sum(row['combined_gate'] for row in gate_flags)}/{len(gate_flags)}")
        print(f"  Acoustic gate: {sum(row['acoustic_gate'] for row in gate_flags)}/{len(gate_flags)}")
        if decoder_hypothesis_info:
            print(f"  Unstable dec.: {sum(row['unstable_decoding'] for row in gate_flags)}/{len(gate_flags)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
