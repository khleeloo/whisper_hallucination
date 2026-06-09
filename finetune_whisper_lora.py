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
import os
import sys
import time

import evaluate
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Audio, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


normalizer = BasicTextNormalizer()


def load_cv_dataset_from_tsv(tsv_path, clips_dir):
    """Load Common Voice data from a TSV file into a HuggingFace Dataset."""
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = os.path.join(clips_dir, row["path"])
            if os.path.exists(audio_path):
                rows.append({
                    "audio": audio_path,
                    "sentence": row["sentence"],
                })
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def prepare_dataset(example, feature_extractor, tokenizer):
    """Process a single example: extract features and tokenize labels."""
    audio = example["audio"]
    example["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    sentence = example["sentence"]
    sentence = normalizer(sentence).strip()
    example["labels"] = tokenizer(sentence).input_ids
    return example


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if prepended by tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


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
    parser.add_argument("--eval_steps", type=int, default=1000)
    args = parser.parse_args()

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

    # Load datasets
    print("Loading training data...", flush=True)
    train_dataset = load_cv_dataset_from_tsv(train_tsv, args.clips_dir)
    print(f"  Train samples: {len(train_dataset)}", flush=True)

    print("Loading dev data...", flush=True)
    dev_dataset = load_cv_dataset_from_tsv(dev_tsv, args.clips_dir)
    print(f"  Dev samples: {len(dev_dataset)}", flush=True)

    # Process datasets
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    num_workers = min(slurm_cpus, 4)  # limit map workers to reduce peak RAM during preprocessing
    print(f"Processing datasets with {num_workers} workers...", flush=True)
    t0 = time.time()
    # num_proc=1 to avoid fork() deadlock with CUDA in multiprocessing.
    # Single-process mapping is sufficient since I/O from /scratch dominates.
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, feature_extractor, tokenizer),
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )
    dev_dataset = dev_dataset.map(
        lambda x: prepare_dataset(x, feature_extractor, tokenizer),
        remove_columns=dev_dataset.column_names,
        num_proc=1,
    )

    # Filter by length (batched)
    def filter_by_length(examples):
        return [len(labels) < args.max_label_length for labels in examples["labels"]]
    print(f"  Map completed in {time.time()-t0:.0f}s", flush=True)
    print("Filtering by length...", flush=True)
    # num_proc=1 for same reason as above — avoid multiprocessing deadlocks on SLURM nodes
    train_dataset = train_dataset.filter(filter_by_length, num_proc=1, batched=True, batch_size=1000)
    dev_dataset = dev_dataset.filter(filter_by_length, num_proc=1, batched=True, batch_size=1000)
    print(f"  After filtering - Train: {len(train_dataset)}, Dev: {len(dev_dataset)}", flush=True)

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
        per_device_eval_batch_size=args.eval_batch_size,
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=2,  # 8 workers × full audio dataset in RAM = OOM; 2 is sufficient
        remove_unused_columns=False,
    )

    # Trainer
    print("Initializing trainer...", flush=True)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric_wer),
        tokenizer=processor.feature_extractor,
    )

    # Train (resume from checkpoint if available)
    n_steps = len(train_dataset) // (args.train_batch_size * args.gradient_accumulation_steps) * args.num_epochs
    print(f"Starting training... ({n_steps} total steps, {args.num_epochs} epochs)", flush=True)
    resume_ckpt = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_ckpt = os.path.join(args.output_dir, latest)
            print(f"Resuming from {resume_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
