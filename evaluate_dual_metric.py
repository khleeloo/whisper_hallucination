"""
Evaluate Whisper models with dual-metric framework:
  1. Lexical accuracy (WAcc = 1 - WER)
  2. Sequence plausibility (LM-based sentence probability)
  3. Repetition analysis (n-gram repetition counts)

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
    elif perturb_type == "none":
        pass
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    # Clip to valid range
    waveform = torch.clamp(waveform, -1.0, 1.0)
    return waveform


def transcribe_batch(model, processor, audio_paths, perturb_type="none",
                     perturb_amplitude=0.0, perturb_duration=0.0,
                     device="cuda", batch_size=16):
    """Transcribe audio files, optionally with perturbation."""
    hypotheses = []

    for batch_start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[batch_start:batch_start + batch_size]
        input_features_list = []

        for path in batch_paths:
            waveform, sample_rate = torchaudio.load(path)
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Apply perturbation
            waveform = apply_perturbation(
                waveform, sample_rate, perturb_type, perturb_amplitude, perturb_duration
            )

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            features = processor.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features[0]
            input_features_list.append(features)

        # Pad and batch
        input_features = torch.stack(input_features_list).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=225,
                language="en",
                task="transcribe",
            )

        transcriptions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        hypotheses.extend(transcriptions)

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Transcribed {min(batch_start + batch_size, len(audio_paths))}/{len(audio_paths)}")

    return hypotheses


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
    lm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
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
            prob_score = np.exp(-nll)
            batch_scores.append(prob_score)

        scores.extend(batch_scores)

        if (batch_start // batch_size) % 20 == 0:
            print(f"  Scored {min(batch_start + batch_size, len(texts))}/{len(texts)}")

    lm_model.cpu()
    del lm_model
    torch.cuda.empty_cache()

    return scores


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

    # Perturbation args
    parser.add_argument("--perturb_type", type=str, default="none",
                        choices=["none", "onset_noise", "full_noise"])
    parser.add_argument("--perturb_amplitude", type=float, default=0.0)
    parser.add_argument("--perturb_duration", type=float, default=0.0,
                        help="Duration of onset noise in seconds")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    perturb_tag = args.perturb_type
    if args.perturb_type != "none":
        perturb_tag = f"{args.perturb_type}_amp{args.perturb_amplitude}_dur{args.perturb_duration}"

    print(f"=== Evaluation: {args.config_name} | Perturbation: {perturb_tag} ===")

    # Load model
    print("Loading model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model = model.to(device)
    model.eval()

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

    # Per-sample WER for joint analysis
    per_sample_wer = []
    for h, r in zip(norm_hyps, norm_refs):
        if len(r) > 0:
            sample_wer = metric_wer.compute(predictions=[h], references=[r])
            per_sample_wer.append(sample_wer)
        else:
            per_sample_wer.append(1.0)
    per_sample_wacc = [1.0 - w for w in per_sample_wer]

    # --- Metric 2: Sequence Plausibility (LM scoring) ---
    print("\nComputing sequence plausibility...")
    hyp_plausibility = compute_lm_perplexity(norm_hyps, model_name=args.lm_model, device=device)
    ref_plausibility = compute_lm_perplexity(norm_refs, model_name=args.lm_model, device=device)

    # Normalized plausibility (hypothesis / reference), clipped to [0, 1]
    norm_plausibility = []
    for hp, rp in zip(hyp_plausibility, ref_plausibility):
        if rp > 0:
            norm_plausibility.append(min(hp / rp, 1.0))
        else:
            norm_plausibility.append(0.0)

    avg_plausibility = np.mean(norm_plausibility)
    avg_raw_plausibility = np.mean(hyp_plausibility)
    print(f"  Avg normalized plausibility: {avg_plausibility:.4f}")
    print(f"  Avg raw plausibility: {avg_raw_plausibility:.4f}")

    # --- Metric 3: Repetition Analysis ---
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

    # --- Summary ---
    results = {
        "config": args.config_name,
        "perturbation": perturb_tag,
        "n_samples": len(samples),
        "wer": round(wer, 4),
        "wacc": round(wacc, 4),
        "avg_normalized_plausibility": round(avg_plausibility, 4),
        "avg_raw_plausibility": round(avg_raw_plausibility, 4),
        "sentences_with_bigram_repeats": sentences_with_reps_2,
        "sentences_with_trigram_repeats": sentences_with_reps_3,
        "sentences_with_4gram_repeats": sentences_with_reps_4,
    }

    # Save results
    results_path = os.path.join(args.output_dir, f"results_{args.config_name}_{perturb_tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save per-sample details for analysis
    details_path = os.path.join(args.output_dir, f"details_{args.config_name}_{perturb_tag}.tsv")
    with open(details_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["reference", "hypothesis", "wacc", "plausibility", "norm_plausibility",
                         "2gram_reps", "3gram_reps", "4gram_reps"])
        for i in range(len(norm_hyps)):
            reps = compute_repetitions(norm_hyps[i])
            writer.writerow([
                norm_refs[i], norm_hyps[i],
                f"{per_sample_wacc[i]:.4f}",
                f"{hyp_plausibility[i]:.4f}",
                f"{norm_plausibility[i]:.4f}",
                reps["2gram_repeats"], reps["3gram_repeats"], reps["4gram_repeats"],
            ])
    print(f"Per-sample details saved to {details_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Config: {args.config_name} | Perturbation: {perturb_tag}")
    print(f"  WAcc:          {wacc:.4f}")
    print(f"  Plausibility:  {avg_plausibility:.4f}")
    print(f"  Bigram reps:   {sentences_with_reps_2}")
    print(f"  Trigram reps:  {sentences_with_reps_3}")
    print(f"  4-gram reps:   {sentences_with_reps_4}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
