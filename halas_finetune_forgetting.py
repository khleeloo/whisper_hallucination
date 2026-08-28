#!/usr/bin/env python3
"""Paired human test of fine-tuning-induced Whisper hallucination on HALAS.

HALAS generated Whisper-large-v3 predictions with the OpenAI whisper package,
not Hugging Face Transformers. Therefore the released HALAS transcript is the
pretrained baseline. We randomly sample HALAS test audio, generate only the
clean LoRA-fine-tuned hypothesis, then create a blinded paired annotation set
containing both transcripts for each audio. The same annotators/protocol can
thus judge pretrained and fine-tuned outputs without a decoding-stack confound.
"""
from __future__ import annotations

import argparse
import json
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
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-10000"
)
DEFAULT_OUT = Path(
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/halas_finetune_forgetting"
)


def download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "halas-ft-forgetting/2.0"})
    with urllib.request.urlopen(req, timeout=180) as response, path.open("wb") as handle:
        handle.write(response.read())


def official_hallucination(row: pd.Series) -> bool:
    h, _loop, halluc_loop = span_labels(row.get(f"{MODEL}_hallucination_json", "[]"))
    return bool(h or halluc_loop)


def get_reference(row: pd.Series) -> str:
    corrected = row.get("corrected_reference_text", "")
    if norm(corrected) and norm(corrected) != "inaudible":
        return str(corrected)
    return str(row.get("e22_reference_text", ""))


def load_halas_test(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / f"HALAS_dataset_{HALAS_COMMIT[:8]}.csv"
    if not csv_path.exists():
        download(HALAS_URL, csv_path)
    wide = pd.read_csv(csv_path)
    pred_col = f"{MODEL}_prediction"
    required = {"audio_id", "split", pred_col, f"{MODEL}_hallucination_json"}
    missing = sorted(required - set(wide.columns))
    if missing:
        raise ValueError(f"HALAS CSV missing required columns: {missing}")
    test = wide[wide["split"].astype(str).str.lower().eq("test")].copy()
    test["halas_prediction"] = test[pred_col].fillna("").astype(str)
    test = test[test["halas_prediction"].map(norm).ne("")].copy()
    test["halas_official_hallucination"] = test.apply(official_hallucination, axis=1)
    test["reference"] = test.apply(get_reference, axis=1)
    return test.reset_index(drop=True)


def _audio_numpy(audio_obj):
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
    return {x for x in {text, stem, text.lower(), stem.lower()} if x}


def build_earnings_index(dataset) -> dict[str, int]:
    index: dict[str, int] = {}
    ambiguous: set[str] = set()
    for i, row in enumerate(dataset):
        seg = str(row.get("segment_id", "")).strip()
        file_id = str(row.get("file_id", "")).strip()
        keys = set()
        for value in (seg, file_id, f"{seg}_{file_id}", f"{file_id}_{seg}"):
            keys |= _id_variants(value)
        for key in keys:
            if key in index and index[key] != i:
                ambiguous.add(key)
            else:
                index[key] = i
    for key in ambiguous:
        index.pop(key, None)
    return index


def resolve_audio_index(audio_id, index: dict[str, int]) -> int | None:
    for key in _id_variants(audio_id):
        if key in index:
            return index[key]
    return None


def prepare_audio(sample: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    print("Loading distil-whisper/earnings22 (chunked/test)...", flush=True)
    earnings = load_dataset("distil-whisper/earnings22", "chunked", split="test")
    index = build_earnings_index(earnings)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows, unresolved = [], []
    for _, row in sample.iterrows():
        audio_idx = resolve_audio_index(row["audio_id"], index)
        if audio_idx is None:
            unresolved.append(str(row["audio_id"]))
            continue
        source = earnings[audio_idx]
        audio, sr = _audio_numpy(source["audio"])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["audio_id"]))
        path = audio_dir / f"{safe_id}.wav"
        if not path.exists():
            sf.write(path, audio, sr)
        item = row.to_dict()
        item["audio_path"] = str(path)
        item["earnings22_index"] = int(audio_idx)
        rows.append(item)
    if unresolved:
        (output_dir / "unresolved_audio_ids.txt").write_text("\n".join(unresolved) + "\n")
    print(f"Resolved sampled audio: {len(rows)}/{len(sample)}", flush=True)
    return pd.DataFrame(rows)


def load_finetuned_model(base_model: str, adapter: str, device: str):
    adapter_path = Path(adapter)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {adapter_path}")
    processor = WhisperProcessor.from_pretrained(base_model, language="en", task="transcribe")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(adapter_path)).to(device)
    model.eval()
    model.config.forced_decoder_ids = None
    return model, processor


def transcribe_paths(model, processor, paths: list[str], device: str, batch_size: int) -> list[str]:
    hypotheses: list[str] = []
    for start in range(0, len(paths), batch_size):
        arrays = []
        for path in paths[start:start + batch_size]:
            audio, sr = sf.read(path, always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import torchaudio
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio).unsqueeze(0), sr, 16000
                ).squeeze(0).numpy()
            arrays.append(audio)
        feats = processor.feature_extractor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feats.input_features.to(device)
        attention_mask = feats.attention_mask.to(device)
        if device.startswith("cuda"):
            input_features = input_features.to(torch.float16)
        with torch.no_grad():
            ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                max_new_tokens=225,
                language="en",
                task="transcribe",
                repetition_penalty=1.0,
            )
        hypotheses.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return hypotheses


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


def exact_mcnemar(base, ft):
    base = np.asarray(base, dtype=int)
    ft = np.asarray(ft, dtype=int)
    introduced = int(((base == 0) & (ft == 1)).sum())
    resolved = int(((base == 1) & (ft == 0)).sum())
    n = introduced + resolved
    p = float(binomtest(min(introduced, resolved), n=n, p=0.5).pvalue) if n else 1.0
    return introduced, resolved, p


def bootstrap_difference(base, ft, n_boot: int, seed: int):
    diff = np.asarray(ft, dtype=float) - np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(diff), size=len(diff))
        draws[i] = diff[idx].mean()
    return float(diff.mean()), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def command_generate(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = load_halas_test(out)
    n = min(args.sample_size, len(test))
    sampled = test.sample(n=n, random_state=args.seed).copy()
    sampled = prepare_audio(sampled, out)
    if len(sampled) != n:
        raise RuntimeError(f"Resolved only {len(sampled)}/{n} sampled HALAS audios")
    sampled = sampled.sort_values("audio_id").reset_index(drop=True)
    sampled["pair_id"] = [f"halas_pair_{i:04d}" for i in range(len(sampled))]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Decoding fine-tuned adapter: {args.adapter}", flush=True)
    model, processor = load_finetuned_model(args.base_model, args.adapter, device)
    sampled["finetuned_prediction"] = transcribe_paths(
        model, processor, sampled["audio_path"].tolist(), device, args.batch_size
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    sampled.to_csv(out / "halas_paired_source.csv", index=False)

    private_rows, public_rows = [], []
    for row in sampled.itertuples(index=False):
        for condition, transcript in (
            ("pretrained_halas", row.halas_prediction),
            ("finetuned", row.finetuned_prediction),
        ):
            annotation_id = f"{row.pair_id}_{'A' if condition == 'pretrained_halas' else 'B'}"
            private_rows.append({
                "annotation_id": annotation_id,
                "pair_id": row.pair_id,
                "condition": condition,
                "audio_id": row.audio_id,
                "audio_path": row.audio_path,
                "transcript": transcript,
                "halas_official_hallucination": (
                    int(row.halas_official_hallucination) if condition == "pretrained_halas" else ""
                ),
            })
            public_rows.append({
                "annotation_id": annotation_id,
                "pair_id": row.pair_id,
                "audio_path": row.audio_path,
                "transcript": transcript,
                "hallucination": "",
            })

    private = pd.DataFrame(private_rows)
    public = pd.DataFrame(public_rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    private.to_csv(out / "annotation_private_manifest.csv", index=False)
    public.to_csv(out / "annotation_blinded.csv", index=False)

    summary = {
        "halas_commit": HALAS_COMMIT,
        "halas_whisper_inference": "OpenAI whisper commit dd985ac; language=en; other parameters default",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "halas_test_whisper_v3_n": int(len(test)),
        "paired_audio_n": int(len(sampled)),
        "annotation_rows": int(len(public)),
        "sample_seed": int(args.seed),
        "protocol": "same audio; released HALAS Whisper-v3 transcript vs clean fine-tuned transcript; both blindly reannotated",
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Blinded annotations: {out / 'annotation_blinded.csv'}", flush=True)


def command_analyze(args):
    out = Path(args.output_dir)
    manifest = pd.read_csv(out / "annotation_private_manifest.csv")
    labels = pd.read_csv(args.annotations)
    required = {"annotation_id", "hallucination"}
    if not required.issubset(labels.columns):
        raise ValueError("Annotations require columns: annotation_id,hallucination")
    labels = labels[["annotation_id", "hallucination"]].copy()
    labels["label"] = labels["hallucination"].map(parse_binary)
    labels = labels[labels["label"].notna()].copy()
    # Supports one or multiple annotators: majority vote per blinded transcript.
    votes = labels.groupby("annotation_id")["label"].mean().rename("vote_mean")
    merged = manifest.merge(votes, on="annotation_id", how="inner")
    merged["human_hallucination"] = (merged["vote_mean"] >= 0.5).astype(int)
    paired = merged.pivot(index="pair_id", columns="condition", values="human_hallucination").dropna()
    if not {"pretrained_halas", "finetuned"}.issubset(paired.columns) or paired.empty:
        raise ValueError("No complete pretrained/fine-tuned annotation pairs")
    base = paired["pretrained_halas"].astype(int).to_numpy()
    ft = paired["finetuned"].astype(int).to_numpy()
    introduced, resolved, p = exact_mcnemar(base, ft)
    delta, ci = bootstrap_difference(base, ft, args.bootstrap, args.seed)

    baseline = merged[merged["condition"].eq("pretrained_halas")].copy()
    baseline["official"] = pd.to_numeric(baseline["halas_official_hallucination"], errors="coerce")
    baseline = baseline[baseline["official"].notna()]
    agreement = float((baseline["official"].astype(int) == baseline["human_hallucination"]).mean()) if len(baseline) else None

    result = {
        "n_paired": int(len(paired)),
        "pretrained_reannotated_hallucination_rate": float(base.mean()),
        "finetuned_hallucination_rate": float(ft.mean()),
        "paired_risk_difference_ft_minus_pretrained": delta,
        "paired_risk_difference_95pct_bootstrap_ci": ci,
        "nonhallucinated_to_hallucinated": introduced,
        "hallucinated_to_nonhallucinated": resolved,
        "mcnemar_exact_p": p,
        "pretrained_reannotation_vs_HALAS_official_agreement": agreement,
    }
    merged.to_csv(out / "human_annotation_merged.csv", index=False)
    paired.reset_index().to_csv(out / "paired_human_results.csv", index=False)
    (out / "paired_human_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    gen.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    gen.add_argument("--output_dir", default=str(DEFAULT_OUT))
    gen.add_argument("--sample_size", type=int, default=300)
    gen.add_argument("--seed", type=int, default=20260828)
    gen.add_argument("--batch_size", type=int, default=8)
    gen.add_argument("--device", default=None)
    gen.set_defaults(func=command_generate)

    ana = sub.add_parser("analyze")
    ana.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ana.add_argument("--annotations", required=True)
    ana.add_argument("--bootstrap", type=int, default=10000)
    ana.add_argument("--seed", type=int, default=20260828)
    ana.set_defaults(func=command_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
