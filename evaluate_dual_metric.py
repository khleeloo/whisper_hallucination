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


def compute_avg_logprobs_from_generate(gen_out):
    """Compute mean generated-token log probability for each sequence."""
    sequences = getattr(gen_out, "sequences", None)
    scores = list(getattr(gen_out, "scores", []) or [])
    if sequences is None:
        return []
    if not scores:
        return [0.0] * int(sequences.shape[0])

    num_steps = min(len(scores), int(sequences.shape[1]))
    if num_steps <= 0:
        return [0.0] * int(sequences.shape[0])

    generated_tokens = sequences[:, -num_steps:]
    token_logprobs = []
    for step_idx, step_scores in enumerate(scores[-num_steps:]):
        step_logprobs = torch.nn.functional.log_softmax(step_scores, dim=-1)
        step_tokens = generated_tokens[:, step_idx].unsqueeze(1).to(step_logprobs.device)
        token_logprobs.append(step_logprobs.gather(1, step_tokens).squeeze(1))

    stacked = torch.stack(token_logprobs, dim=1)
    return stacked.mean(dim=1).detach().cpu().tolist()


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
    }


def _open_threshold_file(path, mode):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def load_gate_thresholds(path):
    """Load gate thresholds from JSON or JSON.GZ."""
    with _open_threshold_file(path, "rt") as handle:
        payload = json.load(handle)
    return {
        "T_logprob": float(payload["T_logprob"]),
        "T_compression": float(payload["T_compression"]),
    }


def save_gate_thresholds(thresholds, path):
    """Save gate thresholds to JSON or JSON.GZ."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "T_logprob": float(thresholds["T_logprob"]),
        "T_compression": float(thresholds["T_compression"]),
    }
    with _open_threshold_file(path, "wt") as handle:
        json.dump(payload, handle, indent=2)


def apply_gate_signals(avg_logprob, compression_ratio, fourgram_rep_count, thresholds):
    """Apply decoder-only hallucination gate signals for one utterance."""
    low_logprob = bool(np.isfinite(avg_logprob) and float(avg_logprob) < float(thresholds["T_logprob"]))
    high_compression = bool(
        np.isfinite(compression_ratio)
        and float(compression_ratio) > float(thresholds["T_compression"])
    )
    fourgram_repetition = int(fourgram_rep_count or 0) >= 1
    return {
        "avg_logprob_only": low_logprob,
        "compression_ratio_only": high_compression,
        "fourgram_repetition_only": fourgram_repetition,
        "combined_gate": bool(low_logprob or high_compression or fourgram_repetition),
    }


def summarize_gate_ablation(flags, hallucination_like, wer, bleu):
    """Summarize gate filtering against evaluation-only labels and metrics."""
    gate_names = [
        "avg_logprob_only",
        "compression_ratio_only",
        "fourgram_repetition_only",
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
                     device="cuda", batch_size=16, return_decoder_signals=False):
    """Transcribe audio files, optionally with perturbation."""
    hypotheses = []
    avg_logprobs = []

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

        with torch.no_grad():
            if return_decoder_signals:
                gen_out = model.generate(
                    input_features["input_features"],
                    attention_mask=input_features.get("attention_mask", None),
                    max_new_tokens=225,
                    language="en",
                    task="transcribe",
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                predicted_ids = gen_out.sequences
                avg_logprobs.extend(compute_avg_logprobs_from_generate(gen_out))
            else:
                predicted_ids = model.generate(
                    input_features["input_features"],
                    attention_mask=input_features.get("attention_mask", None),
                    max_new_tokens=225,
                    language="en",
                    task="transcribe",
                )

        transcriptions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        hypotheses.extend(transcriptions)

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Transcribed {min(batch_start + batch_size, len(audio_paths))}/{len(audio_paths)}")

    if return_decoder_signals:
        return hypotheses, avg_logprobs
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

    mean_wacc = float(np.mean(per_sample_wacc)) if per_sample_wacc else 0.0
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
        key=lambda r: (-r["wacc"], r["rep_total"], -r["plausibility"], r["idx"]),
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
            r["wacc"],
            -r["plausibility"],
            r["idx"],
        ),
    )[:num_examples]

    print("\nQualitative examples:")
    print(f"  Thresholds: mean WAcc={mean_wacc:.4f}, mean plausibility={mean_plausibility:.4f}")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    perturb_tag = args.perturb_type
    if args.perturb_type != "none":
        perturb_tag = f"{args.perturb_type}_amp{args.perturb_amplitude}_dur{args.perturb_duration}"

    print(f"=== Evaluation: {args.config_name} | Perturbation: {perturb_tag} ===")

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
    hypotheses = transcribe_batch(
        model, processor, audio_paths,
        perturb_type=args.perturb_type,
        perturb_amplitude=args.perturb_amplitude,
        perturb_duration=args.perturb_duration,
        device=device,
        batch_size=args.batch_size,
    )

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

    # Paper hallucination-like criterion: above-average sentence probability
    # and below-average WAcc within this evaluation set.
    hallucination_like = [
        bool((np_score > avg_plausibility) and (sample_wacc < avg_sample_wacc))
        for np_score, sample_wacc in zip(norm_plausibility, per_sample_wacc)
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
        f"wacc<{avg_sample_wacc:.4f}"
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
        "hallucination_wacc_threshold": round(finite_float(avg_sample_wacc), 4),
        "hallucination_plausibility_threshold": round(finite_float(avg_plausibility), 4),
        "mean_bleu": round(finite_float(mean_bleu), 4),
        "sentences_with_bigram_repeats": int(sentences_with_reps_2),
        "sentences_with_trigram_repeats": int(sentences_with_reps_3),
        "sentences_with_4gram_repeats": int(sentences_with_reps_4),
    }

    # Save results
    results_path = os.path.join(args.output_dir, f"results_{args.config_name}_{perturb_tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {results_path}")

    # Save per-sample details for analysis
    details_path = os.path.join(args.output_dir, f"details_{args.config_name}_{perturb_tag}.tsv")
    with open(details_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "reference", "hypothesis", "wer", "wacc", "plausibility",
            "norm_plausibility", "hallucination_like",
            "hallucination_wacc_threshold", "hallucination_plausibility_threshold",
            "2gram_reps", "3gram_reps", "4gram_reps",
        ])
        for i in range(len(norm_hyps)):
            reps = compute_repetitions(norm_hyps[i])
            writer.writerow([
                norm_refs[i], norm_hyps[i],
                f"{per_sample_wer[i]:.4f}",
                f"{per_sample_wacc[i]:.4f}",
                f"{hyp_plausibility[i]:.4f}",
                f"{norm_plausibility[i]:.4f}",
                int(hallucination_like[i]),
                f"{avg_sample_wacc:.4f}",
                f"{avg_plausibility:.4f}",
                reps["2gram_repeats"], reps["3gram_repeats"], reps["4gram_repeats"],
            ])
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
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
