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
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

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


def _normalize_config_token(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def validate_requested_model_dir(model_dir, config_name):
    """Catch common run-name mixups before loading an adapter."""
    model_path = Path(model_dir).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    config_token = _normalize_config_token(config_name)
    path_tokens = [_normalize_config_token(part) for part in model_path.parts]

    if config_token in path_tokens:
        return str(model_path)

    # Also allow config names that combine adjacent path components, e.g.
    # rr_64pct_checkpoint-4000 for .../rr_64pct/checkpoint-4000.
    for start in range(len(path_tokens)):
        combined = ""
        for token in path_tokens[start:]:
            combined = token if not combined else f"{combined}_{token}"
            if combined == config_token:
                return str(model_path)
            if len(combined) > len(config_token):
                break

    if config_token not in path_tokens:
        raise ValueError(
            "Model path does not match --config_name. "
            f"config_name={config_name!r}, model_dir={str(model_path)!r}. "
            f"Expected one path component, or adjacent components, to equal {config_token!r}."
        )


def validate_adapter_files(model_dir):
    adapter_config = os.path.join(model_dir, "adapter_config.json")
    adapter_safetensors = os.path.join(model_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(model_dir, "adapter_model.bin")

    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"Missing adapter_config.json in resolved model_dir: {model_dir}")
    if not (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)):
        raise FileNotFoundError(
            "Missing adapter weights in resolved model_dir: "
            f"{model_dir} (expected adapter_model.safetensors or adapter_model.bin)"
        )


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


def select_shard(samples, shard_id=0, num_shards=1):
    """Return a deterministic, disjoint strided shard of samples."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"shard_id must satisfy 0 <= shard_id < num_shards, got {shard_id}/{num_shards}"
        )
    if num_shards == 1:
        return list(samples)
    return list(samples)[shard_id::num_shards]


def resolve_output_suffix(output_suffix=None, shard_id=0, num_shards=1):
    """Resolve the suffix appended to output filenames for sharded runs."""
    if output_suffix:
        if os.path.basename(output_suffix) != output_suffix:
            raise ValueError(f"output_suffix must not contain path separators: {output_suffix!r}")
        return output_suffix
    if num_shards > 1:
        return f"_shard{shard_id:02d}-of-{num_shards:02d}"
    return ""


def resolve_output_paths(output_dir, config_name, output_suffix="", eval_mode="normal"):
    """Return per-utterance CSV and summary JSON paths for an eval mode."""
    if eval_mode not in {"normal", "gated"}:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    mode_suffix = "_gated" if eval_mode == "gated" else ""
    csv_path = os.path.join(
        output_dir,
        f"per_utterance_{config_name}{output_suffix}{mode_suffix}.csv",
    )
    summary_path = os.path.join(
        output_dir,
        f"summary_{config_name}{output_suffix}{mode_suffix}.json",
    )
    return csv_path, summary_path


# --- ASR Inference ---


def load_audio_features(path, feature_extractor):
    """Load one audio file and return Whisper input features."""
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features[0]


def transcribe_batch(
    model,
    processor,
    audio_paths,
    device="cuda",
    batch_size=16,
    audio_num_workers=0,
    return_decoder_signals=False,
):
    """Transcribe a batch of audio files with a Whisper model."""
    hypotheses = []
    avg_logprobs = []
    feature_extractor = processor.feature_extractor
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[batch_start:batch_start + batch_size]
        if audio_num_workers > 0 and len(batch_paths) > 1:
            with ThreadPoolExecutor(max_workers=audio_num_workers) as executor:
                input_features_list = list(
                    executor.map(lambda p: load_audio_features(p, feature_extractor), batch_paths)
                )
        else:
            input_features_list = [load_audio_features(path, feature_extractor) for path in batch_paths]

        input_features = torch.stack(input_features_list).to(device).to(torch.float16)
        # Derive attention mask from non-zero frames (Whisper pad=0, eos same as pad)
        attention_mask = (input_features.abs().sum(dim=(1, 2)) > 0).to(device)

        with torch.no_grad():
            if return_decoder_signals:
                gen_out = model.generate(
                    input_features,
                    attention_mask=attention_mask,
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

    if return_decoder_signals:
        return hypotheses, avg_logprobs
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
        # Paper definition: WAcc = 1 - WER. Do not clamp insertion-heavy
        # utterances; negative values are valid under this definition.
        wacc_val = 1.0 - wer_val

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


def compute_lm_scores_cached(texts, model, tokenizer, device="cuda", batch_size=8):
    """
    Compute sentence probability scores using an already-loaded causal LM.
    Caller handles model loading/unloading.
    Returns list of sentence_score = exp(-avg_token_nll).
    """

    scores = []
    nlls = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        # Filter out empty texts
        valid_indices = []
        valid_texts = []
        for idx, text in enumerate(batch_texts):
            text = text.strip()
            if text:
                valid_indices.append(idx)
                valid_texts.append(text)

        # Initialize scores for all positions in the batch
        batch_scores = [1e-8] * len(batch_texts)
        batch_nlls = [20.0] * len(batch_texts)

        if valid_texts:
            # Batch-tokenize: pad to longest in batch
            encodings = tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            encodings = {k: v.to(model.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])
                nll_batch = outputs.loss.item()

            # Cross-entropy loss is averaged over all non-pad tokens in the batch
            # Split to per-sequence NLL for each valid text
            per_seq_nlls = []
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            logits = outputs.logits

            for i in range(len(valid_texts)):
                seq_len = (encodings["attention_mask"][i] == 1).sum().item()
                if seq_len <= 1:
                    per_seq_nlls.append(20.0)
                    continue
                shift_logits = logits[i, :seq_len - 1, :]
                shift_labels = encodings["input_ids"][i, 1:seq_len]
                nll_val = loss_fct(shift_logits, shift_labels).mean().item()
                per_seq_nlls.append(nll_val)

            for idx, nll_val in zip(valid_indices, per_seq_nlls):
                batch_scores[idx] = np.exp(-nll_val)
                batch_nlls[idx] = nll_val

        scores.extend(batch_scores)
        nlls.extend(batch_nlls)

        if (batch_start // batch_size) % 10 == 0:
            print(f"    Scored {min(batch_start + batch_size, len(texts))}/{len(texts)}", flush=True)

    return scores, nlls


# --- Metric 6: N-gram Repetition ---


# --- Metric 7: BLEU ---


def compute_bleu_scores(hypotheses, references):
    """Compute sentence-level BLEU-4 with smoothing via sacrebleu."""
    import sacrebleu

    results = []
    for hyp, ref in zip(hypotheses, references):
        hyp_norm = normalize_text(hyp)
        ref_norm = normalize_text(ref)

        if not hyp_norm.strip():
            results.append(0.0)
            continue

        bleu_val = sacrebleu.sentence_bleu(
            hyp_norm, [ref_norm], smooth_method="exp"
        ).score
        # sacrebleu returns 0-100 scale; normalize to 0-1
        results.append(bleu_val / 100.0)

    return results


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


def select_plausibility_scores(lm_scores, n_items):
    """Return the plausibility vector used for evaluation-only hall-like labels."""
    if "gpt2" in lm_scores:
        return lm_scores["gpt2"]["norm_scores"], "gpt2"
    if lm_scores:
        lm_name = next(iter(lm_scores))
        return lm_scores[lm_name]["norm_scores"], lm_name
    return [0.0] * n_items, "none"


def _shorten_for_log(text, max_chars=220):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def print_qualitative_examples(
    references,
    hypotheses,
    wer_results,
    rep_results,
    lm_scores,
    num_examples=3,
):
    """Print compact healthy/unhealthy examples for each evaluation run."""
    if not hypotheses or num_examples <= 0:
        return

    primary_lm = next(reversed(lm_scores), None) if lm_scores else None
    fluencies = (
        lm_scores[primary_lm]["norm_scores"]
        if primary_lm
        else [0.0] * len(hypotheses)
    )
    mean_wacc = float(np.mean([r["wacc"] for r in wer_results]))
    mean_fluency = float(np.mean(fluencies)) if fluencies else 0.0

    rows = []
    for idx, (ref, hyp, wer_row, rep_row, fluency) in enumerate(
        zip(references, hypotheses, wer_results, rep_results, fluencies)
    ):
        rep_total = (
            rep_row["bigram_rep_count"]
            + rep_row["trigram_rep_count"]
            + rep_row["fourgram_rep_count"]
        )
        hall_like = wer_row["wacc"] < mean_wacc and fluency > mean_fluency
        rows.append({
            "idx": idx,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer_row["wer"],
            "wacc": wer_row["wacc"],
            "fluency": fluency,
            "rep_total": rep_total,
            "hall_like": hall_like,
        })

    healthy = sorted(
        rows,
        key=lambda r: (-r["wacc"], r["rep_total"], -r["fluency"], r["idx"]),
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
            -r["fluency"],
            r["idx"],
        ),
    )[:num_examples]

    lm_label = primary_lm or "none"
    print("\nQualitative examples:", flush=True)
    print(f"  Thresholds: mean WAcc={mean_wacc:.4f}, mean fluency ({lm_label})={mean_fluency:.4f}", flush=True)

    for title, examples in [("Healthy", healthy), ("Unhealthy", unhealthy)]:
        print(f"  {title} examples:", flush=True)
        for rank, row in enumerate(examples, start=1):
            print(
                f"    {rank}. idx={row['idx']} WAcc={row['wacc']:.4f} "
                f"WER={row['wer']:.4f} fluency={row['fluency']:.4f} "
                f"reps={row['rep_total']} hall_like={int(row['hall_like'])}",
                flush=True,
            )
            print(f"       REF: {_shorten_for_log(row['reference'])}", flush=True)
            print(f"       HYP: {_shorten_for_log(row['hypothesis'])}", flush=True)


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
    parser.add_argument("--audio_num_workers", type=int, default=0,
                        help="Threads per transcription process for audio load/resample/feature extraction")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for cosine similarity")
    parser.add_argument("--lm_models", type=str, nargs="+",
                        default=["gpt2", "Qwen/Qwen3-1.7B"],
                        help="Language models for fluency scoring (weak first, then strong)")
    parser.add_argument("--lm_batch_size", type=int, default=4,
                        help="Batch size for causal-LM fluency scoring")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Zero-based shard id for process-level data parallel eval")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of eval shards")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="Suffix appended to output CSV/JSON filenames")
    parser.add_argument("--eval_mode", choices=["normal", "gated"], default="normal",
                        help="normal keeps historical outputs; gated adds decoder-only gate outputs")
    parser.add_argument("--gate_thresholds_path", type=str, default=None,
                        help="Gated mode only: load gate thresholds from JSON or JSON.GZ")
    parser.add_argument("--calibrate_gate", action="store_true",
                        help="Gated mode only: calibrate gate thresholds from this eval run")
    parser.add_argument("--save_gate_thresholds_path", type=str, default=None,
                        help="Gated mode only: save calibrated gate thresholds to JSON or JSON.GZ")
    parser.add_argument("--skip_lm_scoring", action="store_true",
                        help="Skip LM scoring (for quick WER-only runs)")
    parser.add_argument("--skip_cosine", action="store_true",
                        help="Skip cosine similarity computation")
    parser.add_argument("--num_log_examples", type=int, default=3,
                        help="Number of healthy/unhealthy examples to print per run")

    args = parser.parse_args()
    gated_mode = args.eval_mode == "gated"
    if not gated_mode and (args.gate_thresholds_path or args.calibrate_gate or args.save_gate_thresholds_path):
        parser.error("Gate threshold options require --eval_mode gated")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        output_suffix = resolve_output_suffix(args.output_suffix, args.shard_id, args.num_shards)
        select_shard([], args.shard_id, args.num_shards)
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        sys.exit(1)

    print(f"=== Whisper Validation Eval: {args.config_name} ===", flush=True)
    try:
        args.model_dir = validate_requested_model_dir(args.model_dir, args.config_name)
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        sys.exit(1)
    print(f"  Model: {args.model_dir}", flush=True)
    print(f"  Test set: {args.test_tsv}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Eval mode: {args.eval_mode}", flush=True)
    print(f"  LM models: {args.lm_models}", flush=True)
    if args.num_shards > 1:
        print(f"  Shard: {args.shard_id + 1}/{args.num_shards} ({output_suffix})", flush=True)

    # --- Resolve model directory ---
    model_dir = args.model_dir
    try:
        validate_adapter_files(model_dir)
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        sys.exit(1)
    print(f"  Resolved adapter: {model_dir}", flush=True)

    # --- Load Whisper model ---
    print("\nLoading Whisper model...", flush=True)
    t0 = time.time()
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
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
    total_samples = len(samples)
    samples = select_shard(samples, args.shard_id, args.num_shards)
    if args.num_shards > 1:
        print(f"  Test samples: {total_samples} total, {len(samples)} in this shard", flush=True)
    else:
        print(f"  Test samples: {len(samples)}", flush=True)

    audio_paths = [s["audio_path"] for s in samples]
    references = [s["reference"] for s in samples]
    utt_ids = [s["utt_id"] for s in samples]

    # --- Transcribe ---
    print("Transcribing...", flush=True)
    t0 = time.time()
    if gated_mode:
        hypotheses, avg_logprobs = transcribe_batch(
            model,
            processor,
            audio_paths,
            device=device,
            batch_size=args.batch_size,
            audio_num_workers=args.audio_num_workers,
            return_decoder_signals=True,
        )
    else:
        hypotheses = transcribe_batch(
            model,
            processor,
            audio_paths,
            device=device,
            batch_size=args.batch_size,
            audio_num_workers=args.audio_num_workers,
            return_decoder_signals=False,
        )
        avg_logprobs = None
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

        # Load each LM once and score both hyps + refs
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for lm_name in args.lm_models:
            lm_short = lm_name.split("/")[-1]
            print(f"\n=== LM Scoring: {lm_name} ===", flush=True)

            # Load LM once
            print(f"  Loading LM: {lm_name} ...", flush=True)
            t0 = time.time()
            _dtype = torch.float16 if "cuda" in device else torch.float32
            load_kwargs = {"dtype": _dtype}
            tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=True)
            lm_model = AutoModelForCausalLM.from_pretrained(
                lm_name, trust_remote_code=True, **load_kwargs
            )
            lm_model = lm_model.to(device)
            lm_model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Loaded in {time.time() - t0:.0f}s", flush=True)

            # Score hypotheses (reuse loaded model)
            print("  Scoring hypotheses...", flush=True)
            hyp_scores, hyp_nlls = compute_lm_scores_cached(
                norm_hyps, lm_model, tokenizer, device=device, batch_size=args.lm_batch_size
            )

            # Score references (reuse loaded model)
            print("  Scoring references...", flush=True)
            ref_scores, ref_nlls = compute_lm_scores_cached(
                norm_refs, lm_model, tokenizer, device=device, batch_size=args.lm_batch_size
            )

            # Free LM
            del lm_model
            torch.cuda.empty_cache()

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

    # --- BLEU ---
    print("\nComputing BLEU scores...", flush=True)
    bleu_scores = compute_bleu_scores(hypotheses, references)
    mean_bleu = np.mean(bleu_scores)
    print(f"  Mean BLEU: {mean_bleu:.4f}", flush=True)

    # --- Repetition ---
    print("\nComputing repetition metrics...", flush=True)
    rep_results = [compute_repetition_metrics(h) for h in hypotheses]
    mean_bigram = np.mean([r["bigram_rep_count"] for r in rep_results])
    mean_trigram = np.mean([r["trigram_rep_count"] for r in rep_results])
    mean_fourgram = np.mean([r["fourgram_rep_count"] for r in rep_results])
    print(f"  Mean bigram reps: {mean_bigram:.2f}, trigram: {mean_trigram:.2f}, fourgram: {mean_fourgram:.2f}",
          flush=True)

    gate_thresholds = None
    gate_threshold_source = None
    compression_ratios = None
    gate_flags = None
    plausibility_scores = None
    plausibility_source = None
    hallucination_like = None
    gate_ablation = None
    if gated_mode:
        print("\nComputing decoder-only gate signals...", flush=True)
        compression_ratios = [compute_compression_ratio(text) for text in hypotheses]
        if args.gate_thresholds_path:
            gate_thresholds = load_gate_thresholds(args.gate_thresholds_path)
            gate_threshold_source = args.gate_thresholds_path
        else:
            gate_thresholds = calibrate_gate_thresholds(avg_logprobs, compression_ratios)
            gate_threshold_source = "current_run_calibration"
            if not args.calibrate_gate:
                print("  No gate thresholds supplied; self-calibrating from this run.", flush=True)
        if args.save_gate_thresholds_path:
            save_gate_thresholds(gate_thresholds, args.save_gate_thresholds_path)
            print(f"  Saved gate thresholds: {args.save_gate_thresholds_path}", flush=True)

        gate_flags = [
            apply_gate_signals(
                avg_logprobs[i],
                compression_ratios[i],
                rep_results[i]["fourgram_rep_count"],
                gate_thresholds,
            )
            for i in range(len(hypotheses))
        ]
        plausibility_scores, plausibility_source = select_plausibility_scores(lm_scores, len(hypotheses))
        mean_plausibility = float(np.mean(plausibility_scores)) if plausibility_scores else 0.0
        hallucination_like = [
            bool(wer_results[i]["wacc"] < mean_wacc and plausibility_scores[i] > mean_plausibility)
            for i in range(len(hypotheses))
        ]
        gate_ablation = summarize_gate_ablation(
            gate_flags,
            hallucination_like,
            [row["wer"] for row in wer_results],
            bleu_scores,
        )
        print(
            "  Gate thresholds: "
            f"T_logprob={gate_thresholds['T_logprob']:.4f}, "
            f"T_compression={gate_thresholds['T_compression']:.4f}",
            flush=True,
        )
        print(
            f"  Gate flagged: {sum(row['combined_gate'] for row in gate_flags)}/{len(gate_flags)}",
            flush=True,
        )

    print_qualitative_examples(
        references,
        hypotheses,
        wer_results,
        rep_results,
        lm_scores,
        num_examples=args.num_log_examples,
    )

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
    bleu_cols = ["bleu"]

    rep_cols = [
        "bigram_rep_count", "trigram_rep_count", "fourgram_rep_count",
        "has_bigram_rep", "has_trigram_rep", "has_fourgram_rep",
    ]

    # Decoded text columns
    text_cols = ["decoded_text_raw", "decoded_text_normalized"]

    all_columns = base_columns + wer_detail_cols + cosine_cols + lm_cols + bleu_cols + rep_cols + text_cols
    if gated_mode:
        gate_cols = [
            "condition", "perturbation", "avg_logprob", "compression_ratio",
            "gate_avg_logprob_only", "gate_compression_ratio_only",
            "gate_fourgram_repetition_only", "gate_flagged",
            "plausibility_gpt2_norm", "hallucination_like",
        ]
        all_columns = all_columns + gate_cols

    output_path, summary_path = resolve_output_paths(
        args.output_dir, args.config_name, output_suffix, args.eval_mode
    )
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
                "bleu": bleu_scores[i],
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

            if gated_mode:
                row.update({
                    "condition": args.noise_condition,
                    "perturbation": output_suffix or args.noise_ratio,
                    "avg_logprob": avg_logprobs[i],
                    "compression_ratio": compression_ratios[i],
                    "gate_avg_logprob_only": gate_flags[i]["avg_logprob_only"],
                    "gate_compression_ratio_only": gate_flags[i]["compression_ratio_only"],
                    "gate_fourgram_repetition_only": gate_flags[i]["fourgram_repetition_only"],
                    "gate_flagged": gate_flags[i]["combined_gate"],
                    "plausibility_gpt2_norm": plausibility_scores[i],
                    "hallucination_like": hallucination_like[i],
                })

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
    print(f"  BLEU:   {mean_bleu:.4f}")
    print(f"  Bigram reps:  {mean_bigram:.2f}")
    print(f"  Trigram reps: {mean_trigram:.2f}")
    print(f"  Fourgram reps: {mean_fourgram:.2f}")
    if gated_mode:
        print(f"  Gate flagged: {sum(row['combined_gate'] for row in gate_flags)}/{len(gate_flags)}")
    print(f"{'=' * 60}")

    # Save summary JSON
    summary = {
        "config_name": args.config_name,
        "noise_condition": args.noise_condition,
        "noise_ratio": args.noise_ratio,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "output_suffix": output_suffix,
        "total_samples": total_samples,
        "requested_model_dir": args.model_dir,
        "resolved_model_dir": model_dir,
        "n_samples": len(hypotheses),
        "mean_wer": float(mean_wer),
        "mean_wacc": float(mean_wacc),
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
        "mean_bleu": float(mean_bleu),
        "mean_bigram_rep": float(mean_bigram),
        "mean_trigram_rep": float(mean_trigram),
        "mean_fourgram_rep": float(mean_fourgram),
    }
    for lm_short, lm_data in lm_scores.items():
        summary[f"mean_normalized_score_{lm_short}"] = float(np.mean(lm_data["norm_scores"]))

    if gated_mode:
        summary.update({
            "eval_mode": args.eval_mode,
            "mean_avg_logprob": float(np.mean(avg_logprobs)) if avg_logprobs else 0.0,
            "mean_compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 0.0,
            "gate_thresholds": gate_thresholds,
            "gate_threshold_source": gate_threshold_source,
            "plausibility_source": plausibility_source,
            "hallucination_like_rate": float(np.mean(hallucination_like)) if hallucination_like else 0.0,
            "gate_flag_rate": float(np.mean([row["combined_gate"] for row in gate_flags])) if gate_flags else 0.0,
            "gate_ablation": gate_ablation,
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
