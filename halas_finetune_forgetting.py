#!/usr/bin/env python3
"""Human-grounded test of fine-tuning-induced Whisper hallucination.

Experiment:
1. Load the official HALAS test split and keep whisper_large_v3.
2. Recover the corresponding Earnings22 audio.
3. Decode each item with HF pretrained Whisper-large-v3 and with the clean
   Common-Voice LoRA adapter.
4. Measure normalized transcript reproduction against HALAS's stored pretrained
   Whisper prediction. Existing HALAS human labels are only reused for exact
   reproduced items.
5. Draw a deterministic sample for blinded human annotation of the fine-tuned
   hypotheses.
6. After annotation, run a paired McNemar test and paired-bootstrap CI for the
   change in human hallucination rate.

This script deliberately does not use an automatic hallucination proxy as the
primary endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from peft import PeftModel
from scipy.stats import binomtest
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from halas_external_validation import HALAS_COMMIT, HALAS_URL, norm, span_labels

MODEL = "whisper_large_v3"
DEFAULT_BASE_MODEL = "openai/whisper-large-v3"
DEFAULT_ADAPTER = Path(
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-14000"
)
DEFAULT_OUT = Path(
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/halas_finetune_forgetting"
)


def download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "halas-ft-forgetting/1.0"})
    with urllib.request.urlopen(req, timeout=180) as response, path.open("wb") as handle:
        handle.write(response.read())


def human_hallucination_from_row(row: pd.Series) -> bool:
    h, _loop, halluc_loop = span_labels(row.get(f"{MODEL}_hallucination_json", "[]"))
    return bool(h or halluc_loop)


def get_reference(row: pd.Series) -> str:
    corrected = row.get("corrected_reference_text", "")
    if norm(corrected) and norm(corrected) != "inaudible":
        return str(corrected)
    return str(row.get("e22_reference_text", ""))


def _audio_numpy(audio_obj):
    """Support both datasets Audio dicts and newer AudioDecoder objects."""
    if isinstance(audio_obj, dict):
        return np.asarray(audio_obj["array"], dtype=np.float32), int(audio_obj["sampling_rate"])
    samples = audio_obj.get_all_samples()
    data = samples.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=0)
    return data.squeeze(), int(samples.sample_rate)


def _id_variants(value) -> set[str]:
    text = str(value).strip()
    stem = Path(text).stem
    variants = {text, stem, text.lower(), stem.lower()}
    return {v for v in variants if v}


def build_earnings_index(dataset) -> dict[str, int]:
    """Map HALAS-style IDs to Earnings22 row indices with several safe aliases."""
    index: dict[str, int] = {}
    ambiguous: set[str] = set()
    for i, row in enumerate(dataset):
        seg = str(row.get("segment_id", "")).strip()
        file_id = str(row.get("file_id", "")).strip()
        keys = set()
        keys |= _id_variants(seg)
        keys |= _id_variants(file_id)
        keys |= _id_variants(f"{seg}_{file_id}")
        keys |= _id_variants(f"{file_id}_{seg}")
        for key in keys:
            if key in index and index[key] != i:
                ambiguous.add(key)
            else:
                index[key] = i
    for key in ambiguous:
        index.pop(key, None)
    return index


def resolve_audio_index(audio_id, earnings_index: dict[str, int]) -> int | None:
    for key in _id_variants(audio_id):
        if key in earnings_index:
            return earnings_index[key]
    return None


def load_halas_test(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / f"HALAS_dataset_{HALAS_COMMIT[:8]}.csv"
    if not csv_path.exists():
        download(HALAS_URL, csv_path)
    wide = pd.read_csv(csv_path)
    required = {
        "audio_id",
        "split",
        f"{MODEL}_prediction",
        f"{MODEL}_hallucination_json",
    }
    missing = sorted(required - set(wide.columns))
    if missing:
        raise ValueError(f"HALAS CSV missing required columns: {missing}")
    test = wide[wide["split"].astype(str).str.lower().eq("test")].copy()
    test["halas_prediction"] = test[f"{MODEL}_prediction"].fillna("").astype(str)
    test = test[test["halas_prediction"].map(norm).ne("")].copy()
    test["halas_human_hallucination"] = test.apply(human_hallucination_from_row, axis=1)
    test["reference"] = test.apply(get_reference, axis=1)
    return test.reset_index(drop=True)


def prepare_audio(test: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    print("Loading distil-whisper/earnings22 (chunked/test)...", flush=True)
    earnings = load_dataset("distil-whisper/earnings22", "chunked", split="test")
    earnings_index = build_earnings_index(earnings)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    unresolved = []
    for i, row in test.iterrows():
        audio_idx = resolve_audio_index(row["audio_id"], earnings_index)
        if audio_idx is None:
            unresolved.append(str(row["audio_id"]))
            continue
        source = earnings[audio_idx]
        audio, sr = _audio_numpy(source["audio"])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["audio_id"]))
        path = audio_dir / f"{i:04d}_{safe_id}.wav"
        if not path.exists():
            sf.write(path, audio, sr)
        item = row.to_dict()
        item.update({
            "audio_path": str(path),
            "earnings22_index": int(audio_idx),
            "earnings22_segment_id": str(source.get("segment_id", "")),
            "earnings22_file_id": str(source.get("file_id", "")),
        })
        rows.append(item)

    if unresolved:
        (output_dir / "unresolved_audio_ids.txt").write_text("\n".join(unresolved) + "\n")
    print(f"Resolved audio: {len(rows)}/{len(test)}", flush=True)
    return pd.DataFrame(rows)


def load_model(base_model: str, adapter: str | None, device: str):
    processor = WhisperProcessor.from_pretrained(base_model)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device)
    model.eval()
    return model, processor


def transcribe_paths(model, processor, paths: list[str], device: str, batch_size: int) -> list[str]:
    hypotheses: list[str] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        arrays = []
        for path in batch_paths:
            audio, sr = sf.read(path, always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import torchaudio
                wav = torch.from_numpy(audio).unsqueeze(0)
                audio = torchaudio.functional.resample(wav, sr, 16000).squeeze(0).numpy()
            arrays.append(audio)
        features = processor.feature_extractor(
            arrays, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        if device.startswith("cuda"):
            features = features.to(torch.float16)
        with torch.no_grad():
            ids = model.generate(
                features,
                max_new_tokens=225,
                language="en",
                task="transcribe",
                repetition_penalty=1.0,
            )
        hypotheses.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return hypotheses


def bootstrap_paired_difference(base, ft, n_boot: int, seed: int):
    base = np.asarray(base, dtype=float)
    ft = np.asarray(ft, dtype=float)
    diff = ft - base
    rng = np.random.default_rng(seed)
    n = len(diff)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = diff[idx].mean()
    return float(diff.mean()), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def exact_mcnemar(base, ft):
    base = np.asarray(base, dtype=int)
    ft = np.asarray(ft, dtype=int)
    b = int(((base == 0) & (ft == 1)).sum())  # introduced hallucinations
    c = int(((base == 1) & (ft == 0)).sum())  # resolved hallucinations
    n = b + c
    p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
    return b, c, p


def command_generate(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = load_halas_test(out)
    data = prepare_audio(test, out)
    if data.empty:
        raise RuntimeError("No HALAS audio IDs could be matched to Earnings22")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = data["audio_path"].tolist()

    print("Decoding pretrained Whisper-large-v3...", flush=True)
    raw_model, processor = load_model(args.base_model, None, device)
    data["hf_pretrained_prediction"] = transcribe_paths(
        raw_model, processor, paths, device, args.batch_size
    )
    del raw_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    data["halas_prediction_norm"] = data["halas_prediction"].map(norm)
    data["hf_pretrained_prediction_norm"] = data["hf_pretrained_prediction"].map(norm)
    data["pretrained_exact_reproduction"] = (
        data["halas_prediction_norm"] == data["hf_pretrained_prediction_norm"]
    )
    reproduction_rate = float(data["pretrained_exact_reproduction"].mean())
    print(f"HALAS pretrained transcript exact reproduction: {reproduction_rate:.1%}", flush=True)

    print(f"Decoding fine-tuned adapter: {args.adapter}", flush=True)
    ft_model, ft_processor = load_model(args.base_model, args.adapter, device)
    data["finetuned_prediction"] = transcribe_paths(
        ft_model, ft_processor, paths, device, args.batch_size
    )
    del ft_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    full_path = out / "halas_whisper_v3_pretrained_vs_finetuned.csv"
    data.to_csv(full_path, index=False)

    eligible = data[data["pretrained_exact_reproduction"]].copy()
    if args.sample_size > len(eligible):
        sample_n = len(eligible)
    else:
        sample_n = args.sample_size
    sample = eligible.sample(n=sample_n, random_state=args.seed).copy()
    sample = sample.sort_values("audio_id").reset_index(drop=True)
    sample["item_id"] = [f"halas_ft_{i:04d}" for i in range(len(sample))]

    private_cols = [
        "item_id", "audio_id", "audio_path", "reference", "halas_prediction",
        "halas_human_hallucination", "hf_pretrained_prediction",
        "finetuned_prediction", "pretrained_exact_reproduction",
    ]
    sample[private_cols].to_csv(out / "annotation_private_manifest.csv", index=False)
    public = sample[["item_id", "audio_path", "finetuned_prediction"]].rename(
        columns={"finetuned_prediction": "transcript"}
    )
    public["fine_tuned_hallucination"] = ""
    public.to_csv(out / "annotation_blinded.csv", index=False)

    summary = {
        "halas_commit": HALAS_COMMIT,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "halas_test_whisper_v3_n": int(len(test)),
        "audio_resolved_n": int(len(data)),
        "pretrained_exact_reproduction_n": int(eligible.shape[0]),
        "pretrained_exact_reproduction_rate": reproduction_rate,
        "annotation_sample_n": int(sample_n),
        "sample_seed": int(args.seed),
        "warning": (
            "Existing HALAS pretrained human labels are paired only on exact normalized transcript reproductions. "
            "If reproduction is low, annotate both pretrained and fine-tuned HF outputs instead."
        ),
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Blinded annotation file: {out / 'annotation_blinded.csv'}", flush=True)


def parse_binary(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "hallucination", "hallucinated"}:
        return 1
    if text in {"0", "false", "no", "n", "no hallucination", "not hallucinated"}:
        return 0
    try:
        x = int(float(text))
        return x if x in {0, 1} else np.nan
    except ValueError:
        return np.nan


def command_analyze(args):
    out = Path(args.output_dir)
    private = pd.read_csv(out / "annotation_private_manifest.csv")
    labels = pd.read_csv(args.annotations)
    if "item_id" not in labels or "fine_tuned_hallucination" not in labels:
        raise ValueError("Annotations require columns item_id,fine_tuned_hallucination")
    labels = labels[["item_id", "fine_tuned_hallucination"]].copy()
    labels["ft_h"] = labels["fine_tuned_hallucination"].map(parse_binary)
    merged = private.merge(labels[["item_id", "ft_h"]], on="item_id", how="inner")
    merged = merged[merged["ft_h"].notna()].copy()
    if merged.empty:
        raise ValueError("No valid binary fine-tuned hallucination annotations found")
    base = merged["halas_human_hallucination"].astype(int).to_numpy()
    ft = merged["ft_h"].astype(int).to_numpy()
    introduced, resolved, p = exact_mcnemar(base, ft)
    delta, ci = bootstrap_paired_difference(base, ft, args.bootstrap, args.seed)
    result = {
        "n_paired": int(len(merged)),
        "pretrained_halas_hallucination_rate": float(base.mean()),
        "finetuned_human_hallucination_rate": float(ft.mean()),
        "paired_risk_difference_ft_minus_pretrained": delta,
        "paired_risk_difference_95pct_bootstrap_ci": ci,
        "nonhallucinated_to_hallucinated": introduced,
        "hallucinated_to_nonhallucinated": resolved,
        "mcnemar_exact_p": p,
    }
    merged.to_csv(out / "paired_human_results.csv", index=False)
    (out / "paired_human_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run pretrained/FT inference and create blinded annotation sample")
    gen.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    gen.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    gen.add_argument("--output_dir", default=str(DEFAULT_OUT))
    gen.add_argument("--sample_size", type=int, default=300)
    gen.add_argument("--seed", type=int, default=20260828)
    gen.add_argument("--batch_size", type=int, default=8)
    gen.add_argument("--device", default=None)
    gen.set_defaults(func=command_generate)

    ana = sub.add_parser("analyze", help="Analyze completed blinded human labels")
    ana.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ana.add_argument("--annotations", required=True)
    ana.add_argument("--bootstrap", type=int, default=10000)
    ana.add_argument("--seed", type=int, default=20260828)
    ana.set_defaults(func=command_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
