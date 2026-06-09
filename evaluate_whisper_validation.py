"""
Cross-model validation evaluation for Whisper hallucination experiments.

Evaluates trained LoRA checkpoints with a full suite of per-utterance metrics:
  - WER, WAcc, S/D/I counts (jiwer)
  - Semantic cosine similarity (sentence-transformers)
  - Multi-LM fluency/plausibility scoring
  - N-gram repetition analysis
  - Hallucination-like candidate identification

Output: per_utterance_metrics_whisper.csv

Usage:
    python evaluate_whisper_validation.py \
        --model_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/base/final \
        --base_model openai/whisper-large-v3 \
        --test_tsv /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv \
        --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation \
        --config_name base \
        --noise_condition base \
        --noise_ratio 0.0
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torchaudio
from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Optional imports — loaded lazily
_jiwer_available = False
_sentence_transformers_available = False


def _check_jiwer():
    global _jiwer_available
    if not _jiwer_available:
        try:
            import jiwer
            _jiwer_available = True
        except ImportError:
            raise ImportError("jiwer is required. Install with: pip install jiwer")


def _check_sentence_transformers():
    global _sentence_transformers_available
    if not _sentence_transformers_available:
        try:
            import sentence_transformers  # noqa: F401
            _sentence_transformers_available = True
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )


# --- Text Normalization ---

WHISPER_SPECIAL = re.compile(r"<\|[^|]+\|>")


def normalize_text(text: str) -> str:
    """Normalize text for WER/WAcc computation. Does NOT collapse repeated words."""
    text = text.lower()
    text = WHISPER_SPECIAL.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Data Loading ---


def load_test_data(tsv_path, clips_dir, max_samples=None):
    """Load test samples from a Common Voice-style TSV."""
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            audio_path = os.path.join(clips_dir, row["path"])
            if os.path.exists(audio_path):
                samples.append({
                    "utt_id": row.get("path", f"utt_{i}").replace(".mp3", "").replace(".wav", "").replace("/", "_"),
                    "audio_path": audio_path,
                    "reference": row["sentence"],
                })
    return samples


# --- ASR Inference ---


def transcribe_batch(model, processor, audio_paths, device="cuda", batch_size=16):
    """Transcribe a batch of audio files with a Whisper model."""
    hypotheses = []
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[batch_start:batch_start + batch_size]
        input_features_list = []

        for path in batch_paths:
            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            features = processor.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features[0]
            input_features_list.append(features)

        input_features = torch.stack(input_features_list).to(device).to(torch.float16)
        # Derive attention mask from non-zero frames (Whisper pad=0, eos same as pad)
        attention_mask = (input_features.abs().sum(dim=(1, 2)) > 0).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                max_new_tokens=225,
                language="en",
                task="transcribe",
            )

        transcriptions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        hypotheses.extend(transcriptions)

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Transcribed {min(batch_start + batch_size, len(audio_paths))}/{len(audio_paths)}",
                  flush=True)

    return hypotheses


# --- Metric 1: WER / WAcc ---


def compute_wer_metrics(hypotheses, references):
    """Compute per-utterance WER, WAcc, S, D, I counts."""
    _check_jiwer()
    import jiwer

    results = []
    for hyp, ref in zip(hypotheses, references):
        hyp_norm = normalize_text(hyp)
        ref_norm = normalize_text(ref)

        if len(ref_norm.split()) == 0:
            results.append({
                "wer": 1.0, "wacc": 0.0,
                "s_count": 0, "d_count": 0, "i_count": len(hyp_norm.split()),
                "num_ref_words": 0, "num_hyp_words": len(hyp_norm.split()),
            })
            continue

        measures = jiwer.compute_measures(ref_norm, hyp_norm)
        wer_val = measures["wer"]
        wacc_val = max(0.0, 1.0 - wer_val)

        results.append({
            "wer": wer_val,
            "wacc": wacc_val,
            "s_count": measures.get("substitutions", 0),
            "d_count": measures.get("deletions", 0),
            "i_count": measures.get("insertions", 0),
            "num_ref_words": len(ref_norm.split()),
            "num_hyp_words": len(hyp_norm.split()),
        })

    return results


# --- Metric 2: Cosine Similarity ---


def compute_cosine_similarities(hypotheses, references, model_name="all-MiniLM-L6-v2", device="cuda"):
    """Compute semantic cosine similarity using sentence-transformers."""
    _check_sentence_transformers()
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name} ...", flush=True)
    t0 = time.time()
    embedder = SentenceTransformer(model_name, device=device)
    print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)

    # Normalize texts (keep repeated words — per spec)
    norm_hyps = [normalize_text(h) for h in hypotheses]
    norm_refs = [normalize_text(r) for r in references]

    print(f"Encoding {len(norm_hyps)} hypotheses + references ...", flush=True)
    t0 = time.time()
    emb_hyps = embedder.encode(norm_hyps, show_progress_bar=True, batch_size=64)
    emb_refs = embedder.encode(norm_refs, show_progress_bar=True, batch_size=64)
    print(f"  Encoded in {time.time() - t0:.0f}s", flush=True)

    # Cosine similarity
    similarities = []
    for eh, er in zip(emb_hyps, emb_refs):
        cos_sim = np.dot(eh, er) / (np.linalg.norm(eh) * np.linalg.norm(er) + 1e-8)
        similarities.append(float(cos_sim))

    # Free memory
    del embedder
    torch.cuda.empty_cache()

    return similarities


# --- Metric 3-5: Multi-LM Fluency Scoring ---


def compute_lm_scores(texts, model_name, device="cuda", batch_size=8):
    """
    Compute sentence probability scores using a causal LM.
    Returns list of sentence_score = exp(-avg_token_nll).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading LM: {model_name} ...", flush=True)
    t0 = time.time()

    # Determine dtype and loading strategy
    if "7b" in model_name.lower() or "7B" in model_name:
        torch_dtype = torch.float16
        load_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto"}
    elif "70b" in model_name.lower() or "70B" in model_name:
        torch_dtype = torch.float16
        load_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", "load_in_4bit": True}
    else:
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
        load_kwargs = {"torch_dtype": torch_dtype}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, **load_kwargs
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loaded {model_name} in {time.time() - t0:.0f}s", flush=True)

    scores = []
    nlls = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_scores = []
        batch_nlls = []

        for text in batch_texts:
            text = text.strip()
            if not text:
                batch_scores.append(1e-8)
                batch_nlls.append(20.0)
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = {k: v.to(model.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = model(**input_ids, labels=input_ids["input_ids"])
                nll = outputs.loss.item()

            prob_score = np.exp(-nll)
            batch_scores.append(prob_score)
            batch_nlls.append(nll)

        scores.extend(batch_scores)
        nlls.extend(batch_nlls)

        if (batch_start // batch_size) % 10 == 0:
            print(f"    Scored {min(batch_start + batch_size, len(texts))}/{len(texts)}", flush=True)

    # Free memory
    del model
    torch.cuda.empty_cache()

    return scores, nlls


# --- Metric 6: N-gram Repetition ---


def compute_repetition_metrics(text):
    """
    Compute detailed n-gram repetition metrics.
    repetition_count = sum(count - 1 for each ngram where count > 1)
    has_repetition = repetition_count >= 2
    """
    tokens = normalize_text(text).split()
    results = {}

    for n in [2, 3, 4]:
        if len(tokens) < n:
            results[f"bigram_rep_count" if n == 2 else f"trigram_rep_count" if n == 3 else f"fourgram_rep_count"] = 0
            results[f"has_bigram_rep" if n == 2 else f"has_trigram_rep" if n == 3 else f"has_fourgram_rep"] = False
            continue

        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        rep_count = sum(cnt - 1 for cnt in counts.values() if cnt > 1)

        key_count = f"{['bigram', 'trigram', 'fourgram'][n - 2]}_rep_count"
        key_has = f"has_{['bigram', 'trigram', 'fourgram'][n - 2]}_rep"

        results[key_count] = rep_count
        results[key_has] = rep_count >= 2

    return results


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model validation evaluation for Whisper hallucination experiments"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to LoRA checkpoint (e.g., .../base/final)")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3",
                        help="Base Whisper model name")
    parser.add_argument("--test_tsv", type=str, required=True,
                        help="Path to test TSV")
    parser.add_argument("--clips_dir", type=str, required=True,
                        help="Path to audio clips directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output CSV")
    parser.add_argument("--config_name", type=str, required=True,
                        help="Model config identifier (base, uu, rr, ru, ur)")
    parser.add_argument("--noise_condition", type=str, default="base",
                        help="Noise condition label (base, UU, UR, RR, RU)")
    parser.add_argument("--noise_ratio", type=float, default=0.0,
                        help="Noise ratio used in training (e.g., 0.08)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for debugging)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for transcription")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for cosine similarity")
    parser.add_argument("--lm_models", type=str, nargs="+",
                        default=["gpt2", "Qwen/Qwen3-1.7B"],
                        help="Language models for fluency scoring (weak first, then strong)")
    parser.add_argument("--skip_lm_scoring", action="store_true",
                        help="Skip LM scoring (for quick WER-only runs)")
    parser.add_argument("--skip_cosine", action="store_true",
                        help="Skip cosine similarity computation")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Whisper Validation Eval: {args.config_name} ===", flush=True)
    print(f"  Model: {args.model_dir}", flush=True)
    print(f"  Test set: {args.test_tsv}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  LM models: {args.lm_models}", flush=True)

    # --- Load Whisper model ---
    print("\nLoading Whisper model...", flush=True)
    t0 = time.time()
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model = model.to(device)
    model.eval()

    # Clear forced_decoder_ids set by processor to avoid conflict with task=transcribe
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="en", task="transcribe"
    )
    print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)

    # --- Load test data ---
    print("Loading test data...", flush=True)
    samples = load_test_data(args.test_tsv, args.clips_dir, args.max_samples)
    print(f"  Test samples: {len(samples)}", flush=True)

    audio_paths = [s["audio_path"] for s in samples]
    references = [s["reference"] for s in samples]
    utt_ids = [s["utt_id"] for s in samples]

    # --- Transcribe ---
    print("Transcribing...", flush=True)
    t0 = time.time()
    hypotheses = transcribe_batch(model, processor, audio_paths, device=device,
                                  batch_size=args.batch_size)
    print(f"  Transcribed {len(hypotheses)} utterances in {time.time() - t0:.0f}s", flush=True)

    # Free Whisper model
    model.cpu()
    del model, base_model
    torch.cuda.empty_cache()

    # --- WER / WAcc ---
    print("\nComputing WER metrics...", flush=True)
    wer_results = compute_wer_metrics(hypotheses, references)
    mean_wer = np.mean([r["wer"] for r in wer_results])
    mean_wacc = np.mean([r["wacc"] for r in wer_results])
    print(f"  Mean WER: {mean_wer:.4f}, Mean WAcc: {mean_wacc:.4f}", flush=True)

    # --- Cosine Similarity ---
    if args.skip_cosine:
        cosine_sims = [0.0] * len(hypotheses)
        print("  Skipping cosine similarity.", flush=True)
    else:
        print("\nComputing cosine similarities...", flush=True)
        cosine_sims = compute_cosine_similarities(
            hypotheses, references, model_name=args.embedding_model, device=device
        )
        print(f"  Mean cosine similarity: {np.mean(cosine_sims):.4f}", flush=True)

    # --- LM Fluency Scoring ---
    lm_scores = {}  # {lm_name: {"scores": [...], "nlls": [...], "norm_scores": [...]}}

    if not args.skip_lm_scoring:
        # First compute reference scores once (shared across all LMs)
        norm_refs = [normalize_text(r) for r in references]
        norm_hyps = [normalize_text(h) for h in hypotheses]

        for lm_name in args.lm_models:
            lm_short = lm_name.split("/")[-1]
            print(f"\n=== LM Scoring: {lm_name} ===", flush=True)

            # Score hypotheses
            print("  Scoring hypotheses...", flush=True)
            hyp_scores, hyp_nlls = compute_lm_scores(
                norm_hyps, model_name=lm_name, device=device, batch_size=4
            )

            # Score references
            print("  Scoring references...", flush=True)
            ref_scores, ref_nlls = compute_lm_scores(
                norm_refs, model_name=lm_name, device=device, batch_size=4
            )

            # Normalized scores: hyp_score / ref_score, clipped to [0, 1]
            norm_scores = []
            for hs, rs in zip(hyp_scores, ref_scores):
                ns = hs / (rs + 1e-8)
                ns = min(1.0, max(0.0, ns))
                norm_scores.append(ns)

            lm_scores[lm_short] = {
                "scores": hyp_scores,
                "nlls": hyp_nlls,
                "ref_scores": ref_scores,
                "ref_nlls": ref_nlls,
                "norm_scores": norm_scores,
            }

            mean_norm = np.mean(norm_scores)
            print(f"  Mean normalized score ({lm_short}): {mean_norm:.4f}", flush=True)

    # --- Repetition ---
    print("\nComputing repetition metrics...", flush=True)
    rep_results = [compute_repetition_metrics(h) for h in hypotheses]
    mean_bigram = np.mean([r["bigram_rep_count"] for r in rep_results])
    mean_trigram = np.mean([r["trigram_rep_count"] for r in rep_results])
    mean_fourgram = np.mean([r["fourgram_rep_count"] for r in rep_results])
    print(f"  Mean bigram reps: {mean_bigram:.2f}, trigram: {mean_trigram:.2f}, fourgram: {mean_fourgram:.2f}",
          flush=True)

    # --- Build per-utterance CSV ---
    print("\nBuilding output CSV...", flush=True)

    # Determine column order
    base_columns = [
        "utt_id", "audio_path", "reference", "hypothesis",
        "model_name", "noise_condition", "noise_ratio",
        "wer", "wacc",
    ]

    # WER detail columns
    wer_detail_cols = ["s_count", "d_count", "i_count", "num_ref_words", "num_hyp_words"]

    # Cosine column
    cosine_cols = ["cosine_similarity"]

    # LM columns (dynamically from lm_scores)
    lm_cols = []
    if lm_scores:
        for lm_short in lm_scores:
            lm_cols.extend([
                f"sentence_score_{lm_short}",
                f"nll_{lm_short}",
                f"ref_sentence_score_{lm_short}",
                f"normalized_sentence_score_{lm_short}",
            ])

    # Repetition columns
    rep_cols = [
        "bigram_rep_count", "trigram_rep_count", "fourgram_rep_count",
        "has_bigram_rep", "has_trigram_rep", "has_fourgram_rep",
    ]

    # Decoded text columns
    text_cols = ["decoded_text_raw", "decoded_text_normalized"]

    all_columns = base_columns + wer_detail_cols + cosine_cols + lm_cols + rep_cols + text_cols

    output_path = os.path.join(args.output_dir, f"per_utterance_{args.config_name}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for i in range(len(hypotheses)):
            row = {
                "utt_id": utt_ids[i],
                "audio_path": audio_paths[i],
                "reference": references[i],
                "hypothesis": hypotheses[i],
                "model_name": args.config_name,
                "noise_condition": args.noise_condition,
                "noise_ratio": args.noise_ratio,
                "wer": wer_results[i]["wer"],
                "wacc": wer_results[i]["wacc"],
                "s_count": wer_results[i]["s_count"],
                "d_count": wer_results[i]["d_count"],
                "i_count": wer_results[i]["i_count"],
                "num_ref_words": wer_results[i]["num_ref_words"],
                "num_hyp_words": wer_results[i]["num_hyp_words"],
                "cosine_similarity": cosine_sims[i],
                "bigram_rep_count": rep_results[i]["bigram_rep_count"],
                "trigram_rep_count": rep_results[i]["trigram_rep_count"],
                "fourgram_rep_count": rep_results[i]["fourgram_rep_count"],
                "has_bigram_rep": rep_results[i]["has_bigram_rep"],
                "has_trigram_rep": rep_results[i]["has_trigram_rep"],
                "has_fourgram_rep": rep_results[i]["has_fourgram_rep"],
                "decoded_text_raw": hypotheses[i],
                "decoded_text_normalized": normalize_text(hypotheses[i]),
            }

            # Add LM columns
            for lm_short, lm_data in lm_scores.items():
                row[f"sentence_score_{lm_short}"] = lm_data["scores"][i]
                row[f"nll_{lm_short}"] = lm_data["nlls"][i]
                row[f"ref_sentence_score_{lm_short}"] = lm_data["ref_scores"][i]
                row[f"normalized_sentence_score_{lm_short}"] = lm_data["norm_scores"][i]

            writer.writerow(row)

    print(f"\nSaved per-utterance metrics to: {output_path}", flush=True)
    print(f"  {len(hypotheses)} utterances", flush=True)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Config: {args.config_name} | Condition: {args.noise_condition}")
    print(f"  WAcc:  {mean_wacc:.4f}")
    print(f"  WER:   {mean_wer:.4f}")
    print(f"  Cosine sim: {np.mean(cosine_sims):.4f}")
    for lm_short, lm_data in lm_scores.items():
        print(f"  Fluency ({lm_short}): {np.mean(lm_data['norm_scores']):.4f}")
    print(f"  Bigram reps:  {mean_bigram:.2f}")
    print(f"  Trigram reps: {mean_trigram:.2f}")
    print(f"  Fourgram reps: {mean_fourgram:.2f}")
    print(f"{'=' * 60}")

    # Save summary JSON
    import json
    summary = {
        "config_name": args.config_name,
        "noise_condition": args.noise_condition,
        "noise_ratio": args.noise_ratio,
        "n_samples": len(hypotheses),
        "mean_wer": float(mean_wer),
        "mean_wacc": float(mean_wacc),
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
        "mean_bigram_rep": float(mean_bigram),
        "mean_trigram_rep": float(mean_trigram),
        "mean_fourgram_rep": float(mean_fourgram),
    }
    for lm_short, lm_data in lm_scores.items():
        summary[f"mean_normalized_score_{lm_short}"] = float(np.mean(lm_data["norm_scores"]))

    summary_path = os.path.join(args.output_dir, f"summary_{args.config_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
