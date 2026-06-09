"""
WER-based label-noise filtering for mitigation experiment.

Runs the base Whisper model on a noisy training set's audio, compares the
prediction against the (potentially corrupted) reference label, and keeps
only samples whose WER is below a threshold. The intuition: very high WER
between a strong off-the-shelf ASR and the supplied label is a signal that
the label is corrupted (matches our UU/RR injection patterns).

Output is a new TSV with the same schema, plus a side JSON with stats.

Usage:
    python filter_training_data.py \
        --src_tsv /scratch/.../whisper_hallucination/rr_10/train.tsv \
        --clips_dir /scratch/.../cv-corpus-22.0-2025-06-20/en/clips \
        --dst_tsv /scratch/.../whisper_hallucination/rr_10_filtered/train.tsv \
        --wer_threshold 0.6 \
        --batch_size 16
"""

import argparse
import csv
import json
import os

import evaluate
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def load_rows(tsv_path):
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    return fieldnames, rows


def transcribe(model, processor, audio_paths, clips_dir, device, batch_size):
    hyps = []
    for start in range(0, len(audio_paths), batch_size):
        batch = audio_paths[start:start + batch_size]
        feats = []
        for rel in batch:
            wav, sr = torchaudio.load(os.path.join(clips_dir, rel))
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            f = processor.feature_extractor(wav.squeeze().numpy(), sampling_rate=16000,
                                            return_tensors="pt").input_features[0]
            feats.append(f)
        inputs = torch.stack(feats).to(device)
        # Derive attention mask from non-zero frames (Whisper pad=0, eos same as pad)
        attention_mask = (inputs.abs().sum(dim=(1, 2)) > 0).to(device)
        with torch.no_grad():
            ids = model.generate(inputs, attention_mask=attention_mask,
                                 max_new_tokens=225, language="en", task="transcribe")
        hyps.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        if (start // batch_size) % 20 == 0:
            print(f"  transcribed {min(start + batch_size, len(audio_paths))}/{len(audio_paths)}")
    return hyps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_tsv", required=True)
    p.add_argument("--dst_tsv", required=True)
    p.add_argument("--clips_dir", required=True)
    p.add_argument("--base_model", default="openai/whisper-large-v3")
    p.add_argument("--wer_threshold", type=float, default=0.6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.dst_tsv), exist_ok=True)

    print(f"Loading {args.src_tsv}")
    fieldnames, rows = load_rows(args.src_tsv)
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"  {len(rows)} rows")

    print("Loading base Whisper (no LoRA) for filtering")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                             torch_dtype=torch.float16).to(device).eval()
    # Clear forced_decoder_ids set by processor to avoid conflict with task=transcribe
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    processor = WhisperProcessor.from_pretrained(args.base_model, language="en", task="transcribe")

    audio_paths = [r["path"] for r in rows]
    refs = [r["sentence"] for r in rows]

    print("Transcribing training audio with base model...")
    hyps = transcribe(model, processor, audio_paths, args.clips_dir, device, args.batch_size)

    metric = evaluate.load("wer")
    print("Scoring per-sample WER and filtering...")
    kept = []
    per_sample = []
    for r, h, ref in zip(rows, hyps, refs):
        h_n = normalizer(h).strip()
        r_n = normalizer(ref).strip()
        if not r_n:
            wer = 1.0
        else:
            wer = metric.compute(predictions=[h_n], references=[r_n])
        per_sample.append(wer)
        if wer <= args.wer_threshold:
            kept.append(r)

    print(f"\nKept {len(kept)}/{len(rows)} = {100 * len(kept) / len(rows):.1f}%")

    with open(args.dst_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)
    print(f"Wrote {args.dst_tsv}")

    stats = {
        "src": args.src_tsv,
        "dst": args.dst_tsv,
        "wer_threshold": args.wer_threshold,
        "n_total": len(rows),
        "n_kept": len(kept),
        "kept_fraction": len(kept) / max(1, len(rows)),
        "wer_mean": float(sum(per_sample) / max(1, len(per_sample))),
    }
    with open(args.dst_tsv + ".filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
