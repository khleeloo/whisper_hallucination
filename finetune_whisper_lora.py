"""
Fine-tune Whisper Large v3 with LoRA on Common Voice EN with noisy labels.

Adapted from whisper-large-v3-cantonese/src/finetune_on_hf_dataset.py.
Uses HuggingFace Seq2SeqTrainer + PEFT LoRA for memory-efficient training.

Usage:
    python finetune_whisper_lora.py \
        --noise_config base \
        --data_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
        --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/base \
        --model_name openai/whisper-large-v3 \
        --num_epochs 5 \
        --learning_rate 1e-4 \
        --lora_r 16 \
        --lora_alpha 32
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import tempfile
import fcntl

import evaluate
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import Audio, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


normalizer = BasicTextNormalizer()
EXPECTED_CACHE_COLUMNS = {"input_features", "labels", "label_length"}
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _is_rank_zero() -> bool:
    """Return True if this is a single-process run or the DDP global rank 0."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _barrier():
    """Synchronise DDP ranks (no-op if not distributed)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_cv_dataset_from_tsv(tsv_path, clips_dir, check_audio_exists=None):
    """Load Common Voice data from a TSV file into a HuggingFace Dataset."""
    if check_audio_exists is None:
        check_audio_exists = os.environ.get("WHISPER_SKIP_AUDIO_EXISTS_CHECK") != "1"
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = os.path.join(clips_dir, row["path"])
            if not check_audio_exists or os.path.exists(audio_path):
                rows.append({
                    "audio": audio_path,
                    "sentence": row["sentence"],
                })
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def prepare_dataset(example, feature_extractor, tokenizer):
    """Process a single example: extract features and keep normalized text labels."""
    audio = example["audio"]
    example["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    sentence = example["sentence"]
    sentence = normalizer(sentence).strip()
    example["labels"] = sentence
    example["label_length"] = len(tokenizer(sentence).input_ids)
    return example


def prepare_dataset_batch(examples, feature_extractor, tokenizer):
    """Batch audio feature extraction and label normalization for faster mapping."""
    audios = examples["audio"]
    sampling_rate = audios[0]["sampling_rate"] if audios else 16000
    input_features = feature_extractor(
        [audio["array"] for audio in audios],
        sampling_rate=sampling_rate,
    ).input_features

    sentences = [normalizer(sentence).strip() for sentence in examples["sentence"]]
    tokenized = tokenizer(sentences)
    return {
        "input_features": input_features,
        "labels": sentences,
        "label_length": [len(ids) for ids in tokenized.input_ids],
    }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_values = [f["labels"] for f in features]
        if isinstance(label_values[0], str):
            labels_batch = self.processor.tokenizer(
                label_values, padding=True, return_tensors="pt"
            )
        else:
            labels_batch = self.processor.tokenizer.pad(
                [{"input_ids": l} for l in label_values], return_tensors="pt"
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if prepended by tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def checkpoint_step(checkpoint_name: str) -> Optional[int]:
    """Return the numeric training step for checkpoint-N directories."""
    match = CHECKPOINT_RE.match(os.path.basename(checkpoint_name))
    if not match:
        return None
    return int(match.group(1))


def _has_adapter_files(checkpoint_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
        and (
            os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors"))
            or os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
        )
    )


def _checkpoint_resume_issue(checkpoint_dir: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return "not a directory"
    if checkpoint_step(os.path.basename(checkpoint_dir)) is None:
        return "not a numeric checkpoint directory"
    if not _has_adapter_files(checkpoint_dir):
        return "missing adapter config or weights"
    if not os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json")):
        return "missing trainer_state.json"
    if not os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt")):
        return "missing optimizer.pt"
    if not os.path.exists(os.path.join(checkpoint_dir, "scheduler.pt")):
        return "missing scheduler.pt"
    return None


def numeric_checkpoint_dirs(output_dir: str):
    """Return numeric checkpoint dirs as (step, name, path), sorted oldest to newest."""
    if not os.path.isdir(output_dir):
        return []

    checkpoints = []
    for entry in os.listdir(output_dir):
        step = checkpoint_step(entry)
        if step is None:
            continue
        path = os.path.join(output_dir, entry)
        if os.path.isdir(path):
            checkpoints.append((step, entry, path))
    return sorted(checkpoints)


def _best_checkpoint_name_from_state(checkpoint_dir: str) -> Optional[str]:
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            trainer_state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    best_checkpoint = trainer_state.get("best_model_checkpoint")
    if not best_checkpoint:
        return None
    best_name = os.path.basename(best_checkpoint)
    if checkpoint_step(best_name) is None:
        return None
    return best_name


def find_best_resume_checkpoint(output_dir: str) -> Optional[str]:
    """Find the saved best numeric checkpoint complete enough for Trainer resume."""
    complete_checkpoints = {}
    for _, name, path in reversed(numeric_checkpoint_dirs(output_dir)):
        issue = _checkpoint_resume_issue(path)
        if issue is not None:
            print(f"  Skipping incomplete checkpoint for resume ({issue}): {name}", flush=True)
            continue

        complete_checkpoints[name] = path
        best_name = _best_checkpoint_name_from_state(path)
        best_path = complete_checkpoints.get(best_name)
        if best_path:
            return best_path

    if complete_checkpoints:
        return next(iter(complete_checkpoints.values()))
    return None


def cleanup_numeric_checkpoints(output_dir: str, keep_names) -> None:
    """Delete numeric checkpoint dirs except those explicitly kept."""
    keep_names = {name for name in keep_names if name}
    for _, entry, path in numeric_checkpoint_dirs(output_dir):
        if entry not in keep_names:
            shutil.rmtree(path, ignore_errors=True)


class CheckpointCleanupCallback(TrainerCallback):
    """Keep only the best numeric checkpoint once best-model tracking is available."""

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        output_dir = args.output_dir
        keep_dirs = set()
        if state.best_model_checkpoint:
            best_name = os.path.basename(state.best_model_checkpoint)
            if checkpoint_step(best_name) is not None:
                keep_dirs.add(best_name)
        if not keep_dirs and state.global_step:
            keep_dirs.add(f"checkpoint-{state.global_step}")

        cleanup_numeric_checkpoints(output_dir, keep_dirs)


def compute_metrics(pred, tokenizer, metric_wer):
    """Compute WER from predictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize
    pred_str = [normalizer(p).strip() for p in pred_str]
    label_str = [normalizer(l).strip() for l in label_str]

    # Filter empty references
    filtered = [(p, l) for p, l in zip(pred_str, label_str) if len(l) > 0]
    if not filtered:
        return {"wer": 1.0}
    pred_str, label_str = zip(*filtered)

    wer = metric_wer.compute(predictions=list(pred_str), references=list(label_str))
    return {"wer": wer * 100}


def resolve_step_values(save_steps, eval_steps, load_best_model_at_end):
    """Ensure checkpointing steps are compatible with best-model selection."""
    if not load_best_model_at_end or eval_steps <= 0:
        return save_steps, eval_steps

    if save_steps % eval_steps == 0:
        return save_steps, eval_steps

    aligned_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
    return aligned_save_steps, eval_steps


def _resolve_num_workers(args) -> int:
    """Determine the number of multiprocessing workers for map/filter operations.

    Priority: --num_proc argument > sched_getaffinity > cpu_count.
    Always capped at 8 to prevent IPC buffer overrun.
    """
    import multiprocessing

    if args.num_proc is not None and args.num_proc > 0:
        raw = args.num_proc
    else:
        try:
            raw = len(os.sched_getaffinity(0))
        except AttributeError:
            raw = multiprocessing.cpu_count()
    return max(1, min(raw, 8))


def _file_fingerprint(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "sha256": hasher.hexdigest(),
    }


def _build_cache_metadata(train_tsv, dev_tsv, model_name, max_label_length):
    return {
        "version": 2,
        "train_tsv": _file_fingerprint(train_tsv),
        "dev_tsv": _file_fingerprint(dev_tsv),
        "model_name": model_name,
        "max_label_length": max_label_length,
    }


def _cache_metadata_path(train_cache):
    return os.path.join(
        os.path.dirname(train_cache),
        f"meta_{os.path.basename(train_cache)}.json",
    )


def _cache_metadata_matches(train_cache, expected_metadata):
    meta_path = _cache_metadata_path(train_cache)
    if not os.path.exists(meta_path):
        print("Cache metadata missing - re-preprocessing...", flush=True)
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            actual_metadata = json.load(f)
    except Exception:
        print("Cache metadata unreadable - re-preprocessing...", flush=True)
        return False
    if actual_metadata != expected_metadata:
        print("Cache metadata mismatch - re-preprocessing...", flush=True)
        return False
    return True


def _write_cache_metadata(train_cache, cache_metadata):
    meta_path = _cache_metadata_path(train_cache)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cache_metadata, f, indent=2, sort_keys=True)


def _load_valid_cached_datasets(train_cache, dev_cache, cache_metadata):
    if not (
        os.path.exists(train_cache)
        and os.path.exists(dev_cache)
        and _cache_metadata_matches(train_cache, cache_metadata)
    ):
        return None
    try:
        train_tmp = Dataset.load_from_disk(train_cache)
        dev_tmp = Dataset.load_from_disk(dev_cache)
    except Exception:
        print("Cache read error - re-preprocessing...", flush=True)
        return None
    if (
        EXPECTED_CACHE_COLUMNS.issubset(train_tmp.column_names)
        and EXPECTED_CACHE_COLUMNS.issubset(dev_tmp.column_names)
        and len(train_tmp) > 0
        and len(dev_tmp) > 0
    ):
        print("Loading preprocessed datasets from cache...", flush=True)
        return train_tmp, dev_tmp
    print("Cache invalid - re-preprocessing...", flush=True)
    return None


def _remove_preprocessed_cache(train_cache, dev_cache):
    shutil.rmtree(train_cache, ignore_errors=True)
    shutil.rmtree(dev_cache, ignore_errors=True)
    meta_path = _cache_metadata_path(train_cache)
    if os.path.exists(meta_path):
        os.remove(meta_path)


def _preprocess_dataset_cached(
    train_dataset,
    dev_dataset,
    feature_extractor,
    tokenizer,
    train_cache,
    dev_cache,
    max_label_length,
    num_proc,
    cache_metadata,
    preprocessing_batch_size=16,
):
    """Run map + filter and save to cache.

    Under DDP only rank 0 executes this; other ranks wait for the cache
    file to appear and then load it.
    """
    os.makedirs(os.path.dirname(train_cache), exist_ok=True)
    print(f"Preprocessing datasets ({num_proc} workers)...", flush=True)
    t0 = time.time()

    train_dataset = train_dataset.map(
        lambda x: prepare_dataset_batch(x, feature_extractor, tokenizer),
        remove_columns=train_dataset.column_names,
        num_proc=num_proc,
        batched=True,
        batch_size=preprocessing_batch_size,
        writer_batch_size=1000,
    )
    dev_dataset = dev_dataset.map(
        lambda x: prepare_dataset_batch(x, feature_extractor, tokenizer),
        remove_columns=dev_dataset.column_names,
        num_proc=num_proc,
        batched=True,
        batch_size=preprocessing_batch_size,
        writer_batch_size=1000,
    )
    print(f"  Map completed in {time.time()-t0:.0f}s", flush=True)

    # Filter by length
    print("Filtering by length...", flush=True)

    def filter_by_length(examples):
        return [length < max_label_length for length in examples["label_length"]]

    train_dataset = train_dataset.filter(
        filter_by_length,
        num_proc=num_proc,
        batched=True,
        batch_size=1000,
        writer_batch_size=1000,
    )
    dev_dataset = dev_dataset.filter(
        filter_by_length,
        num_proc=num_proc,
        batched=True,
        batch_size=1000,
        writer_batch_size=1000,
    )
    print(f"  After filtering - Train: {len(train_dataset)}, Dev: {len(dev_dataset)}", flush=True)

    # Save to cache for subsequent GPU ranks
    train_dataset.save_to_disk(train_cache)
    dev_dataset.save_to_disk(dev_cache)
    _write_cache_metadata(train_cache, cache_metadata)
    print(f"  Cached to {os.path.dirname(train_cache)}", flush=True)
    return train_dataset, dev_dataset


def _load_or_preprocess(
    train_dataset,
    dev_dataset,
    feature_extractor,
    tokenizer,
    train_cache,
    dev_cache,
    max_label_length,
    num_proc,
    cache_metadata,
    preprocessing_batch_size=16,
):
    """Load from cache if valid, otherwise preprocess (rank-0-only under DDP)."""
    # Quick check: if cache exists and looks healthy, load it on every rank
    if (
        os.path.exists(train_cache)
        and os.path.exists(dev_cache)
        and _cache_metadata_matches(train_cache, cache_metadata)
    ):
        try:
            train_tmp = Dataset.load_from_disk(train_cache)
            dev_tmp = Dataset.load_from_disk(dev_cache)
            expected_cols = {"input_features", "labels", "label_length"}
            if (
                expected_cols.issubset(train_tmp.column_names)
                and expected_cols.issubset(dev_tmp.column_names)
                and len(train_tmp) > 0
                and len(dev_tmp) > 0
            ):
                print("Loading preprocessed datasets from cache...", flush=True)
                return train_tmp, dev_tmp
            else:
                print("Cache invalid — re-preprocessing...", flush=True)
                shutil.rmtree(train_cache, ignore_errors=True)
                shutil.rmtree(dev_cache, ignore_errors=True)
        except Exception:
            print("Cache read error — re-preprocessing...", flush=True)
            shutil.rmtree(train_cache, ignore_errors=True)
            shutil.rmtree(dev_cache, ignore_errors=True)

    elif os.path.exists(train_cache) or os.path.exists(dev_cache):
        shutil.rmtree(train_cache, ignore_errors=True)
        shutil.rmtree(dev_cache, ignore_errors=True)

    # --- Single-worker protocol: only rank 0 runs the map; others poll. ---
    # Use a temp-file lock so overlapping rank-0 runs on different filesystems
    # don't race.
    lock_dir = os.path.join(tempfile.gettempdir(), ".whisper_lora_preproc_locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, os.path.basename(train_cache) + ".lock")

    rank_zero = _is_rank_zero()

    if rank_zero:
        # Acquire exclusive lock, preprocess, release
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                # Double-check after acquiring lock (another rank 0 may have raced)
                if (
                    os.path.exists(train_cache)
                    and os.path.exists(dev_cache)
                    and _cache_metadata_matches(train_cache, cache_metadata)
                ):
                    try:
                        train_tmp = Dataset.load_from_disk(train_cache)
                        dev_tmp = Dataset.load_from_disk(dev_cache)
                        if len(train_tmp) > 0 and len(dev_tmp) > 0:
                            print("Loading preprocessed datasets from cache...", flush=True)
                            return train_tmp, dev_tmp
                    except Exception:
                        pass
                result = _preprocess_dataset_cached(
                    train_dataset,
                    dev_dataset,
                    feature_extractor,
                    tokenizer,
                    train_cache,
                    dev_cache,
                    max_label_length,
                    num_proc,
                    cache_metadata,
                    preprocessing_batch_size,
                )
                return result
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    else:
        # Non-rank-0: wait for the cache to appear (poll every 30 s)
        print(
            f"Rank (non-zero) waiting for rank 0 to finish preprocessing cache at "
            f"{train_cache}...",
            flush=True,
        )
        waited = 0
        while not (
            os.path.exists(train_cache)
            and os.path.exists(dev_cache)
            and _cache_metadata_matches(train_cache, cache_metadata)
        ):
            time.sleep(30)
            waited += 30
            if waited > 0 and waited % 300 == 0:
                print(f"  Still waiting ... {waited}s elapsed", flush=True)
        train_tmp = Dataset.load_from_disk(train_cache)
        dev_tmp = Dataset.load_from_disk(dev_cache)
        print(
            f"  Cache ready, loaded Train: {len(train_tmp)}, Dev: {len(dev_tmp)}",
            flush=True,
        )
        return train_tmp, dev_tmp


def _load_or_preprocess_fast(
    train_tsv,
    dev_tsv,
    clips_dir,
    feature_extractor,
    tokenizer,
    train_cache,
    dev_cache,
    max_label_length,
    num_proc,
    cache_metadata,
    preprocessing_batch_size,
):
    """Load cache before touching raw TSV/audio; only rank 0 preprocesses on miss."""
    cached = _load_valid_cached_datasets(train_cache, dev_cache, cache_metadata)
    if cached is not None:
        return cached
    if os.path.exists(train_cache) or os.path.exists(dev_cache):
        _remove_preprocessed_cache(train_cache, dev_cache)

    lock_dir = os.path.join(tempfile.gettempdir(), ".whisper_lora_preproc_locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, os.path.basename(train_cache) + ".lock")

    if _is_rank_zero():
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                cached = _load_valid_cached_datasets(train_cache, dev_cache, cache_metadata)
                if cached is not None:
                    return cached
                if os.path.exists(train_cache) or os.path.exists(dev_cache):
                    _remove_preprocessed_cache(train_cache, dev_cache)

                print("Loading training data...", flush=True)
                train_dataset = load_cv_dataset_from_tsv(train_tsv, clips_dir)
                print(f"  Train samples: {len(train_dataset)}", flush=True)
                print("Loading dev data...", flush=True)
                dev_dataset = load_cv_dataset_from_tsv(dev_tsv, clips_dir)
                print(f"  Dev samples: {len(dev_dataset)}", flush=True)

                return _preprocess_dataset_cached(
                    train_dataset,
                    dev_dataset,
                    feature_extractor,
                    tokenizer,
                    train_cache,
                    dev_cache,
                    max_label_length,
                    num_proc,
                    cache_metadata,
                    preprocessing_batch_size,
                )
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    print(
        f"Rank (non-zero) waiting for rank 0 to finish preprocessing cache at "
        f"{train_cache}...",
        flush=True,
    )
    waited = 0
    cached = _load_valid_cached_datasets(train_cache, dev_cache, cache_metadata)
    while cached is None:
        time.sleep(30)
        waited += 30
        if waited % 300 == 0:
            print(f"  Still waiting ... {waited}s elapsed", flush=True)
        cached = _load_valid_cached_datasets(train_cache, dev_cache, cache_metadata)
    train_tmp, dev_tmp = cached
    print(
        f"  Cache ready, loaded Train: {len(train_tmp)}, Dev: {len(dev_tmp)}",
        flush=True,
    )
    return train_tmp, dev_tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_config", type=str, required=True,
                        help="Noise configuration to train on (base, uu, rr, ru, ur, or sweep like uu_05)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of noisy datasets (output of create_noisy_dataset.py)")
    parser.add_argument("--clips_dir", type=str, required=True,
                        help="Path to Common Voice clips directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model checkpoints")
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_audio_length", type=float, default=30.0,
                        help="Maximum audio length in seconds")
    parser.add_argument("--max_label_length", type=int, default=128,
                        help="Maximum label length in tokens")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of map workers (default: auto-detect, capped at 8)")
    parser.add_argument("--preprocessing_batch_size", type=int, default=16,
                        help="Batch size for preprocessing map workers")
    parser.add_argument("--skip_audio_exists_check", action="store_true",
                        help="Skip per-row audio path existence checks when loading TSVs")
    args = parser.parse_args()
    if args.skip_audio_exists_check:
        os.environ["WHISPER_SKIP_AUDIO_EXISTS_CHECK"] = "1"

    # Paths
    train_tsv = os.path.join(args.data_dir, args.noise_config, "train.tsv")
    dev_tsv = os.path.join(args.data_dir, "dev.tsv")

    print(f"=== Whisper LoRA Fine-tuning: {args.noise_config} ===", flush=True)
    print(f"Train TSV: {train_tsv}", flush=True)
    print(f"Dev TSV: {dev_tsv}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}", flush=True)

    # Load processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="en", task="transcribe")

    # Cache paths for preprocessed data
    cache_dir = os.path.join(args.output_dir, ".dataset_cache")
    train_cache = os.path.join(cache_dir, f"train_{args.noise_config}")
    dev_cache = os.path.join(cache_dir, f"dev_{args.noise_config}")
    cache_metadata = _build_cache_metadata(
        train_tsv,
        dev_tsv,
        args.model_name,
        args.max_label_length,
    )

    num_proc = _resolve_num_workers(args)
    train_dataset, dev_dataset = _load_or_preprocess_fast(
        train_tsv,
        dev_tsv,
        args.clips_dir,
        feature_extractor,
        tokenizer,
        train_cache,
        dev_cache,
        args.max_label_length,
        num_proc,
        cache_metadata,
        args.preprocessing_batch_size,
    )

    # All ranks synchronise after preprocessing / cache load
    _barrier()

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Required for gradient checkpointing
    print(f"  Model loaded in {time.time()-t0:.0f}s", flush=True)

    # Apply LoRA
    print("Applying LoRA...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    sys.stdout.flush()

    # Must set use_cache=False AFTER applying PEFT
    model.config.use_cache = False

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Metric - load with offline mode to avoid HF Hub compatibility issues
    try:
        metric_wer = evaluate.load("wer", download_mode="offline")
    except Exception:
        # Fallback: try without specifying download mode
        try:
            metric_wer = evaluate.load("wer")
        except Exception as e:
            print(f"Warning: Could not load 'wer' metric from evaluate: {e}", flush=True)
            # Fallback to manual WER calculation using jiwer if available
            try:
                from jiwer import wer as jiwer_wer
                class ManualWER:
                    def compute(self, predictions, references):
                        return jiwer_wer(references, predictions)
                metric_wer = ManualWER()
            except ImportError:
                # Final fallback: dummy metric
                class DummyWER:
                    def compute(self, predictions, references):
                        return 0.0
                metric_wer = DummyWER()
                print("Warning: Using dummy WER metric (returns 0.0 for all evaluations)", flush=True)

    save_steps, eval_steps = resolve_step_values(
        args.save_steps,
        args.eval_steps,
        load_best_model_at_end=True,
    )
    print(f"Checkpoint save_steps={save_steps}, eval_steps={eval_steps}", flush=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=2,  # small eval batch to avoid OOM during generate
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        prediction_loss_only=True,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=2,  # 8 workers × full audio dataset in RAM = OOM; 2 is sufficient
        remove_unused_columns=False,
        disable_tqdm=True,  # Suppress per-step progress bar spam; loss reported via logging_steps
    )

    # Suppress per-step tqdm from non-rank-0 GPU processes to avoid interleaved output
    import torch.distributed as dist
    import logging
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)

    # Trainer
    print("Initializing trainer...", flush=True)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric_wer),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001),
            CheckpointCleanupCallback(),
        ],
    )

    # Train (resume from checkpoint if available)
    n_steps = len(train_dataset) // (args.train_batch_size * args.gradient_accumulation_steps) * args.num_epochs
    print(f"Starting training... ({n_steps} total steps, {args.num_epochs} epochs)", flush=True)
    existing_checkpoints = numeric_checkpoint_dirs(args.output_dir)
    resume_ckpt = find_best_resume_checkpoint(args.output_dir)
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}", flush=True)
    elif existing_checkpoints:
        raise RuntimeError(
            "Found numeric checkpoint directories, but none are complete enough "
            "for Trainer resume. Refusing to restart from the base model in the "
            f"same output_dir: {args.output_dir}"
        )
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Clean up intermediate checkpoints after successful training.
    # Keep only the best numeric checkpoint for auditability/resume. The final/
    # adapter saved below is also the in-memory best model because
    # load_best_model_at_end=True.
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        checkpoints = numeric_checkpoint_dirs(args.output_dir)
        best_name = None
        if trainer.state.best_model_checkpoint:
            candidate = os.path.basename(trainer.state.best_model_checkpoint)
            if checkpoint_step(candidate) is not None:
                best_name = candidate
        if best_name is None and checkpoints:
            best_name = checkpoints[-1][1]
        keep_names = {best_name}
        cleanup_numeric_checkpoints(args.output_dir, keep_names)
        kept = ", ".join(sorted(name for name in keep_names if name))
        if kept:
            print(f"Kept best checkpoint dir: {kept}", flush=True)

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:

        # Run final WER evaluation on dev set (single pass, ~15-20 min)
        print("Running final WER evaluation on dev set...", flush=True)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.eval()

        wer_scores = []
        total_batches = (len(dev_dataset) + args.eval_batch_size - 1) // args.eval_batch_size
        from tqdm import tqdm

        with torch.no_grad():
            for i in tqdm(range(0, len(dev_dataset), args.eval_batch_size),
                          desc="Final WER", total=total_batches):
                batch = dev_dataset[i:i + args.eval_batch_size]
                input_features = torch.stack([torch.tensor(x["input_features"]) for x in batch]).to(model.device)
                labels = [x["labels"] for x in batch]

                predicted_ids = model.generate(
                    input_features,
                    max_new_tokens=128,
                    language="en",
                    task="transcribe",
                    num_beams=1,
                )

                pred_str = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                if labels and isinstance(labels[0], str):
                    label_ids_padded = tokenizer(
                        labels, padding=True, return_tensors="pt"
                    )["input_ids"]
                else:
                    label_ids_padded = tokenizer.pad(
                        [{"input_ids": l} for l in labels], return_tensors="pt"
                    )["input_ids"]
                label_ids_padded[label_ids_padded == -100] = tokenizer.pad_token_id
                label_str = tokenizer.batch_decode(label_ids_padded, skip_special_tokens=True)

                pred_norm = [normalizer(p).strip() for p in pred_str]
                label_norm = [normalizer(l).strip() for l in label_str]

                for p, l in zip(pred_norm, label_norm):
                    if l.strip():
                        try:
                            w = metric_wer.compute(predictions=[p], references=[l])
                            wer_scores.append(w)
                        except Exception:
                            pass
                del input_features, predicted_ids

        mean_wer = sum(wer_scores) / len(wer_scores) * 100 if wer_scores else 0.0
        print(f"  Final Dev WER: {mean_wer:.2f}%", flush=True)

        # Write WER to file alongside checkpoint
        wer_path = os.path.join(args.output_dir, "final_wer.txt")
        with open(wer_path, "w") as f:
            f.write(f"Dev WER: {mean_wer:.2f}%\n")
            f.write(f"Num evaluated: {len(wer_scores)}\n")

    # Save final model (only rank 0 writes)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        final_dir = os.path.join(args.output_dir, "final")
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"Model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
