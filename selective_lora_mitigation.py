#!/usr/bin/env python3
"""Cheap post-hoc mitigation screen for fine-tuning-induced Whisper hallucination.

No retraining. The same clean Common-Voice LoRA adapter is evaluated with
selective LoRA contribution scaling:
  pretrained      encoder=0.0 decoder=0.0
  finetuned       encoder=1.0 decoder=1.0
  decoder_0.5     encoder=1.0 decoder=0.5
  decoder_0.25    encoder=1.0 decoder=0.25
  decoder_0       encoder=1.0 decoder=0.0
  encoder_0       encoder=0.0 decoder=1.0   (mechanistic control)

Primary use: screen on clean Common Voice plus one matched full-noise condition,
then send only the selected mitigation to HALAS human evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from evaluate_dual_metric import transcribe_batch as transcribe_perturbed_batch
from evaluate_whisper_validation import (
    compute_repetition_metrics,
    compute_wer_metrics,
    load_test_data,
    normalize_text,
)
from mitigation_experiment import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CLIPS_DIR,
    DEFAULT_QWEN_MODEL,
    DEFAULT_TEST_TSV,
    DEFAULT_THRESHOLD_SOURCE_CSV,
    DEFAULT_THRESHOLDS_JSON,
    _default_lm_plausibility,
    _load_whisper_model,
    apply_frozen_hallucination_thresholds,
    load_or_create_frozen_hallucination_thresholds,
    selected_perturbations,
)

DEFAULT_ADAPTER = Path(
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-14000"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/selective_lora_mitigation"
)
DEFAULT_VARIANTS = [
    ("pretrained", 0.0, 0.0),
    ("finetuned", 1.0, 1.0),
    ("decoder_0.5", 1.0, 0.5),
    ("decoder_0.25", 1.0, 0.25),
    ("decoder_0", 1.0, 0.0),
    ("encoder_0", 0.0, 1.0),
]
DEFAULT_PERTURBATIONS = ["none", "full_noise_amp0.5_dur0.0"]


def collect_lora_scalings(model):
    """Capture original PEFT scaling values and classify LoRA modules by Whisper side."""
    entries = []
    unknown = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or not scaling:
            continue
        if ".encoder." in name:
            side = "encoder"
        elif ".decoder." in name:
            side = "decoder"
        else:
            unknown.append(name)
            continue
        for adapter_name, value in scaling.items():
            entries.append((name, module, adapter_name, float(value), side))

    if not entries:
        raise RuntimeError("No PEFT LoRA scaling dictionaries found in the loaded model.")
    if unknown:
        raise RuntimeError(
            "Found LoRA modules outside explicit Whisper encoder/decoder paths: "
            + ", ".join(unknown[:10])
        )
    counts = {
        "encoder": sum(side == "encoder" for *_, side in entries),
        "decoder": sum(side == "decoder" for *_, side in entries),
    }
    if counts["encoder"] == 0 or counts["decoder"] == 0:
        raise RuntimeError(f"Expected LoRA modules on both encoder and decoder, got {counts}")
    return entries, counts


def apply_selective_scaling(entries, encoder_factor: float, decoder_factor: float) -> None:
    """Scale LoRA deltas linearly by modifying each module's PEFT scaling value."""
    for _name, module, adapter_name, original, side in entries:
        factor = encoder_factor if side == "encoder" else decoder_factor
        module.scaling[adapter_name] = original * float(factor)


def score_text_outputs(df: pd.DataFrame) -> pd.DataFrame:
    wer_rows = compute_wer_metrics(df["hypothesis"].tolist(), df["reference"].tolist())
    rep_rows = [compute_repetition_metrics(text) for text in df["hypothesis"]]
    out = df.copy()
    out["WER"] = [row["wer"] for row in wer_rows]
    out["rep3"] = [row["trigram_rep_count"] for row in rep_rows]
    out["rep4"] = [row["fourgram_rep_count"] for row in rep_rows]
    out["rep34"] = ((out["rep3"] > 0) | (out["rep4"] > 0)).astype(int)
    out["prediction_norm"] = [normalize_text(text) for text in out["hypothesis"]]
    return out


def add_qwen_proxy(
    df: pd.DataFrame,
    *,
    device: str,
    qwen_model: str,
    batch_size: int,
    thresholds_json: Path,
    threshold_source_csv: Path,
) -> pd.DataFrame:
    out = df.copy()
    out["qwen_plaus"] = list(
        _default_lm_plausibility(
            out["hypothesis"].tolist(),
            out["reference"].tolist(),
            qwen_model,
            device,
            batch_size,
        )
    )
    thresholds = load_or_create_frozen_hallucination_thresholds(
        thresholds_json, threshold_source_csv
    )
    out["hallucination_like"] = apply_frozen_hallucination_thresholds(
        out, thresholds, plausibility_col="qwen_plaus"
    ).astype(int)
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, perturbation), g in df.groupby(["variant", "perturbation"], sort=False):
        counts = g["prediction_norm"].value_counts(normalize=True)
        rows.append(
            {
                "variant": variant,
                "perturbation": perturbation,
                "encoder_scale": float(g["encoder_scale"].iloc[0]),
                "decoder_scale": float(g["decoder_scale"].iloc[0]),
                "N": int(len(g)),
                "WER": float(g["WER"].mean()),
                "qwen_plaus": float(g["qwen_plaus"].mean()) if "qwen_plaus" in g else np.nan,
                "hallucination_like_rate": (
                    float(g["hallucination_like"].mean()) if "hallucination_like" in g else np.nan
                ),
                "rep34_rate": float(g["rep34"].mean()),
                "top1_mass": float(counts.iloc[0]) if len(counts) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def choose_mitigation(summary: pd.DataFrame, retain_gain: float, stress_label: str) -> Dict[str, object]:
    clean = summary[summary["perturbation"] == "none"].set_index("variant")
    stress = summary[summary["perturbation"] == stress_label].set_index("variant")
    if "pretrained" not in clean.index or "finetuned" not in clean.index:
        return {"selected": None, "reason": "missing pretrained or finetuned clean rows"}
    pre_wer = float(clean.loc["pretrained", "WER"])
    ft_wer = float(clean.loc["finetuned", "WER"])
    gain = pre_wer - ft_wer
    if gain <= 0:
        return {
            "selected": None,
            "reason": "fine-tuning did not improve clean WER on the screening subset",
            "pretrained_clean_WER": pre_wer,
            "finetuned_clean_WER": ft_wer,
        }

    max_wer = pre_wer - retain_gain * gain
    candidates = ["finetuned", "decoder_0.5", "decoder_0.25", "decoder_0"]
    eligible = [
        name for name in candidates
        if name in clean.index and name in stress.index and float(clean.loc[name, "WER"]) <= max_wer
    ]
    if not eligible:
        return {
            "selected": None,
            "reason": "no decoder mitigation retained the requested fraction of clean WER gain",
            "required_max_clean_WER": max_wer,
        }

    if "hallucination_like_rate" in stress.columns and stress.loc[eligible, "hallucination_like_rate"].notna().any():
        ranked = sorted(
            eligible,
            key=lambda name: (
                float(stress.loc[name, "hallucination_like_rate"]),
                float(clean.loc[name, "WER"]),
            ),
        )
        criterion = "lowest stress hallucination-like rate, then clean WER"
    else:
        ranked = sorted(eligible, key=lambda name: (float(stress.loc[name, "WER"]), float(clean.loc[name, "WER"])))
        criterion = "lowest stress WER, then clean WER (Qwen proxy skipped)"

    selected = ranked[0]
    return {
        "selected": selected,
        "criterion": criterion,
        "retain_gain_fraction_required": retain_gain,
        "pretrained_clean_WER": pre_wer,
        "finetuned_clean_WER": ft_wer,
        "fine_tuning_clean_WER_gain": gain,
        "required_max_clean_WER": max_wer,
        "selected_clean_WER": float(clean.loc[selected, "WER"]),
        "selected_stress_WER": float(stress.loc[selected, "WER"]),
        "selected_stress_hallucination_like_rate": (
            float(stress.loc[selected, "hallucination_like_rate"])
            if pd.notna(stress.loc[selected, "hallucination_like_rate"])
            else None
        ),
        "eligible": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--qwen_batch_size", type=int, default=8)
    parser.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--skip_qwen", action="store_true")
    parser.add_argument("--perturbations", nargs="+", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--stress_label", default="full_noise_amp0.5_dur0.0")
    parser.add_argument("--retain_gain", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--thresholds_json", type=Path, default=DEFAULT_THRESHOLDS_JSON)
    parser.add_argument("--threshold_source_csv", type=Path, default=DEFAULT_THRESHOLD_SOURCE_CSV)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_test_data(str(args.test_tsv), str(args.clips_dir), max_samples=args.max_samples)
    if not samples:
        raise RuntimeError("No evaluation samples loaded.")
    audio_paths = [sample["audio_path"] for sample in samples]
    references = [sample["reference"] for sample in samples]
    utterance_ids = [sample["utt_id"] for sample in samples]
    perturbations = selected_perturbations(args.perturbations)

    model, base_model, processor = _load_whisper_model(args.adapter, args.base_model, device)
    entries, counts = collect_lora_scalings(model)
    print(f"LoRA entries: encoder={counts['encoder']} decoder={counts['decoder']}", flush=True)

    frames: List[pd.DataFrame] = []
    try:
        for variant_idx, (variant, enc_scale, dec_scale) in enumerate(DEFAULT_VARIANTS):
            apply_selective_scaling(entries, enc_scale, dec_scale)
            print(f"\n=== {variant}: encoder={enc_scale} decoder={dec_scale} ===", flush=True)
            for perturb_idx, perturbation in enumerate(perturbations):
                # Reset RNG so every variant receives exactly the same random corruption.
                matched_seed = args.seed + perturb_idx
                np.random.seed(matched_seed)
                torch.manual_seed(matched_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(matched_seed)
                hypotheses = transcribe_perturbed_batch(
                    model,
                    processor,
                    audio_paths,
                    perturb_type=perturbation.perturb_type,
                    perturb_amplitude=perturbation.amplitude,
                    perturb_duration=perturbation.duration,
                    device=device,
                    batch_size=args.batch_size,
                    repetition_penalty=1.0,
                )
                frame = pd.DataFrame(
                    {
                        "utterance_id": utterance_ids,
                        "variant": variant,
                        "encoder_scale": enc_scale,
                        "decoder_scale": dec_scale,
                        "perturbation": perturbation.label,
                        "reference": references,
                        "hypothesis": hypotheses,
                        "audio_path": audio_paths,
                    }
                )
                frames.append(score_text_outputs(frame))
    finally:
        model.cpu()
        del model, base_model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.concat(frames, ignore_index=True)
    if args.skip_qwen:
        result["qwen_plaus"] = np.nan
        result["hallucination_like"] = np.nan
    else:
        result = add_qwen_proxy(
            result,
            device=device,
            qwen_model=args.qwen_model,
            batch_size=args.qwen_batch_size,
            thresholds_json=args.thresholds_json,
            threshold_source_csv=args.threshold_source_csv,
        )

    per_path = args.output_dir / "per_utterance_selective_lora.csv"
    result.to_csv(per_path, index=False)
    summary = aggregate(result)
    summary_path = args.output_dir / "summary_selective_lora.csv"
    summary.to_csv(summary_path, index=False)

    selection = choose_mitigation(summary, args.retain_gain, args.stress_label)
    selection.update(
        {
            "adapter": str(args.adapter),
            "max_samples": len(samples),
            "stress_label": args.stress_label,
            "lora_entry_counts": counts,
        }
    )
    (args.output_dir / "selected_mitigation.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n=== Summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== Selection ===", flush=True)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
