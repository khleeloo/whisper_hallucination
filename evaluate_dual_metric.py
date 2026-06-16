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
            prob_score = np.exp(-nll)
            batch_scores.append(prob_score)

        scores.extend(batch_scores)

        if (batch_start // batch_size) % 20 == 0:
            print(f"  Scored {min(batch_start + batch_size, len(texts))}/{len(texts)}")

    lm_model.cpu()
    del lm_model
    torch.cuda.empty_cache()

    return scores


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
                        choices=["none", "onset_noise", "full_noise"])
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
            if rp > 0:
                norm_plausibility.append(min(hp / rp, 1.0))
            else:
                norm_plausibility.append(0.0)

        avg_plausibility = np.mean(norm_plausibility)
        avg_raw_plausibility = np.mean(hyp_plausibility)
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
        (np_score > avg_plausibility) and (sample_wacc < avg_sample_wacc)
        for np_score, sample_wacc in zip(norm_plausibility, per_sample_wacc)
    ]
    hallucination_like_count = sum(hallucination_like)
    hallucination_like_rate = (
        hallucination_like_count / len(hallucination_like)
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
        "n_samples": len(samples),
        "wer": round(wer, 4),
        "wacc": round(wacc, 4),
        "mean_sample_wacc": round(avg_sample_wacc, 4),
        "avg_normalized_plausibility": round(avg_plausibility, 4),
        "avg_raw_plausibility": round(avg_raw_plausibility, 4),
        "hallucination_like_count": hallucination_like_count,
        "hallucination_like_rate": round(hallucination_like_rate, 4),
        "hallucination_wacc_threshold": round(avg_sample_wacc, 4),
        "hallucination_plausibility_threshold": round(avg_plausibility, 4),
        "mean_bleu": round(mean_bleu, 4),
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
