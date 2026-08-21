#!/usr/bin/env python3
"""Run the diagnostic acoustic-stress pipeline on pretrained SeamlessM4T-v2 Large.

This is the cross-architecture replication experiment for the ASR hallucination
paper.  It uses the public Transformers checkpoint
``facebook/seamless-m4t-v2-large`` directly, with no task-specific fine-tuning.

Protocol
--------
1. Decode matched Common Voice DEV/TEST utterances under:
      clean, full-noise 0.50, full-noise 0.75.
2. Recompute WER with the corrected clean normalization.
3. Score every hypothesis with Qwen3-0.6B and GPT-2 plausibility, repetition,
   output concentration, and independent wav2vec2-CTC acoustic support.
4. Freeze model-specific LM plausibility thresholds from clean DEV. Report:
      diagnostic H: WER > clean-DEV mean WER AND LM plausibility > clean-DEV mean;
      strict H:     WER > 0.5 AND LM plausibility > clean-DEV mean.
5. Use stressed DEV examples to calibrate a reference-free wav2vec2 acoustic-
   support gate subject to a minimum clean-DEV coverage constraint.  Tune with
   strict Qwen labels, freeze the threshold, then apply unchanged to held-out
   clean and stressed TEST speech.

The acoustic perturbation, corrected WER, LM plausibility, CTC-support, and gate
selection code are reused from the Whisper experiments so that the target ASR
model is the main experimental change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from acoustic_abstention_mitigation import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_TEST_TSV,
    accepted_mask,
    load_exact_waveform,
    load_manifest,
    resolve_dev_tsv,
    score_ctc_support,
    select_gate_threshold,
    validate_disjoint,
)
from acoustic_grounding_validation import DEFAULT_MODEL_NAME as DEFAULT_WAV2VEC2_MODEL
from clean_wer_rescore import add_clean_wer, normalize_asr_text
from evaluate_whisper_validation import compute_repetition_metrics
from mitigation_experiment import DEFAULT_GPT2_MODEL, DEFAULT_QWEN_MODEL, _default_lm_plausibility

ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_OUTPUT_DIR = ROOT / "seamless_m4t_v2_stress_pipeline"
DEFAULT_MODEL = "facebook/seamless-m4t-v2-large"
DEFAULT_PERTURBATIONS = [
    "none",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]
DEFAULT_STRICT_WER = 0.5

GENERATED_COLUMNS = [
    "split",
    "utterance_id",
    "audio_path",
    "reference",
    "perturbation",
    "hypothesis",
]


def load_seamless(model_name: str, device: str):
    """Load the dedicated Transformers speech-to-text model and processor."""
    try:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    except ImportError as exc:
        raise RuntimeError(
            "The installed transformers version does not provide "
            "SeamlessM4Tv2ForSpeechToText. Update transformers and install "
            "sentencepiece in the llama environment."
        ) from exc

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Loading pretrained S2TT model: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, processor


def decode_manifest_seamless(
    model: object,
    processor: object,
    manifest: pd.DataFrame,
    perturbation: str,
    *,
    device: str,
    batch_size: int,
    base_seed: int,
    tgt_lang: str,
    max_new_tokens: int,
) -> pd.DataFrame:
    hypotheses: List[str] = []
    rows = list(manifest.itertuples(index=False))
    model_dtype = next(model.parameters()).dtype

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        waveforms = [
            load_exact_waveform(row, perturbation, base_seed=base_seed)
            for row in batch
        ]
        inputs = processor(
            audios=waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        moved = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
            elif key == "input_features":
                moved[key] = value.to(device=device, dtype=model_dtype)
            else:
                moved[key] = value.to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **moved,
                tgt_lang=tgt_lang,
                max_new_tokens=max_new_tokens,
            )
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        hypotheses.extend([str(x).strip() for x in decoded])

        done = min(start + len(batch), len(rows))
        if start == 0 or done == len(rows) or (start // batch_size) % 10 == 0:
            print(
                f"  {manifest.iloc[0]['split']} {perturbation}: transcribed {done}/{len(rows)}",
                flush=True,
            )

    out = manifest[["split", "utterance_id", "audio_path", "reference"]].copy()
    out["perturbation"] = perturbation
    out["hypothesis"] = hypotheses
    return out[GENERATED_COLUMNS]


def add_repetition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows = [compute_repetition_metrics(str(h)) for h in out["hypothesis"]]
    out["rep2"] = [int(r.get("bigram_rep_count", 0)) for r in rows]
    out["rep3"] = [int(r.get("trigram_rep_count", 0)) for r in rows]
    out["rep4"] = [int(r.get("fourgram_rep_count", 0)) for r in rows]
    out["rep34"] = (out["rep3"] > 0) | (out["rep4"] > 0)
    return out


def add_dual_lm_plausibility(
    df: pd.DataFrame,
    *,
    device: str,
    qwen_model: str,
    gpt2_model: str,
    batch_size: int,
) -> pd.DataFrame:
    out = df.copy()
    for column, model_name in (("qwen_plaus", qwen_model), ("gpt2_plaus", gpt2_model)):
        print(f"Scoring {len(out):,} rows with {model_name}...", flush=True)
        out[column] = list(
            _default_lm_plausibility(
                out["hypothesis"].astype(str).tolist(),
                out["reference"].astype(str).tolist(),
                model_name,
                device,
                batch_size,
            )
        )
    return out


def derive_thresholds(df: pd.DataFrame, strict_wer: float) -> Dict[str, float]:
    clean = df[(df["split"] == "dev") & (df["perturbation"] == "none")].copy()
    clean = clean[
        clean["valid_reference_cleanwer"].astype(bool)
        & np.isfinite(clean["WER"].astype(float))
    ]
    if clean.empty:
        raise ValueError("No valid clean DEV rows available for threshold derivation")
    return {
        "diagnostic_wer_threshold": float(clean["WER"].astype(float).mean()),
        "strict_wer_threshold": float(strict_wer),
        "qwen_plausibility_threshold": float(clean["qwen_plaus"].astype(float).mean()),
        "gpt2_plausibility_threshold": float(clean["gpt2_plaus"].astype(float).mean()),
        "N_clean_dev_valid": int(len(clean)),
    }


def apply_labels(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    valid = out["valid_reference_cleanwer"].astype(bool) & np.isfinite(out["WER"].astype(float))
    diag_high_wer = valid & (out["WER"].astype(float) > thresholds["diagnostic_wer_threshold"])
    strict_high_wer = valid & (out["WER"].astype(float) > thresholds["strict_wer_threshold"])

    for lm in ("qwen", "gpt2"):
        high_plaus = out[f"{lm}_plaus"].astype(float) > thresholds[f"{lm}_plausibility_threshold"]
        out[f"diag_h_{lm}"] = diag_high_wer & high_plaus
        out[f"strict_h_{lm}"] = strict_high_wer & high_plaus

    out["diag_h_both"] = out["diag_h_qwen"] & out["diag_h_gpt2"]
    out["diag_h_union"] = out["diag_h_qwen"] | out["diag_h_gpt2"]
    out["strict_h_both"] = out["strict_h_qwen"] & out["strict_h_gpt2"]
    out["strict_h_union"] = out["strict_h_qwen"] | out["strict_h_gpt2"]
    return out


def concentration_metrics(hypotheses: Sequence[object]) -> Dict[str, float]:
    normalized = [normalize_asr_text(x) for x in hypotheses]
    nonempty = [x for x in normalized if x]
    n = len(normalized)
    if not nonempty:
        return {
            "empty_rate": 1.0 if n else float("nan"),
            "unique_nonempty_fraction": 0.0,
            "top1_mass": 0.0,
            "top10_mass": 0.0,
        }
    counts = pd.Series(nonempty).value_counts()
    return {
        "empty_rate": float((n - len(nonempty)) / n) if n else float("nan"),
        "unique_nonempty_fraction": float(counts.size / len(nonempty)),
        "top1_mass": float(counts.iloc[0] / len(nonempty)),
        "top10_mass": float(counts.iloc[:10].sum() / len(nonempty)),
    }


def summarize_conditions(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    subset = df[df["split"] == split]
    for condition, g in subset.groupby("perturbation", sort=False):
        valid = g["valid_reference_cleanwer"].astype(bool)
        gv = g[valid]
        row: Dict[str, object] = {
            "split": split,
            "condition": condition,
            "N": int(len(g)),
            "N_valid": int(valid.sum()),
            "WER_mean": float(gv["WER"].mean()) if len(gv) else float("nan"),
            "WER_median": float(gv["WER"].median()) if len(gv) else float("nan"),
            "exact_after_normalization_pct": 100.0 * float((gv["WER"] == 0).mean()) if len(gv) else float("nan"),
            "qwen_plaus_mean": float(g["qwen_plaus"].astype(float).mean()),
            "gpt2_plaus_mean": float(g["gpt2_plaus"].astype(float).mean()),
            "diag_H_qwen_pct": 100.0 * float(g["diag_h_qwen"].astype(bool).mean()),
            "diag_H_gpt2_pct": 100.0 * float(g["diag_h_gpt2"].astype(bool).mean()),
            "strict_H_qwen_pct": 100.0 * float(g["strict_h_qwen"].astype(bool).mean()),
            "strict_H_gpt2_pct": 100.0 * float(g["strict_h_gpt2"].astype(bool).mean()),
            "strict_H_both_pct": 100.0 * float(g["strict_h_both"].astype(bool).mean()),
            "strict_H_union_pct": 100.0 * float(g["strict_h_union"].astype(bool).mean()),
            "rep34_pct": 100.0 * float(g["rep34"].astype(bool).mean()),
            "ctc_support_nll_mean": float(
                g["ctc_support_nll"].replace([np.inf, -np.inf], np.nan).mean()
            ),
            **concentration_metrics(g["hypothesis"].tolist()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def gate_test_summary(test: pd.DataFrame, tau: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for condition, g in test.groupby("perturbation", sort=False):
        accepted = accepted_mask(g["ctc_support_nll"].astype(float), tau)
        row: Dict[str, object] = {
            "condition": condition,
            "N": int(len(g)),
            "coverage": float(accepted.mean()),
            "abstention": float((~accepted).mean()),
            "WER_before": float(g["WER"].mean(skipna=True)),
            "WER_emitted": float(g.loc[accepted, "WER"].mean(skipna=True)) if accepted.any() else float("nan"),
        }
        for kind in ("diag", "strict"):
            for lm in ("qwen", "gpt2"):
                h = g[f"{kind}_h_{lm}"].astype(bool).to_numpy()
                n_h = int(h.sum())
                emitted_h = h & accepted
                row[f"{kind}_{lm}_H_before"] = float(h.mean())
                row[f"{kind}_{lm}_H_after_system"] = float(emitted_h.mean())
                row[f"{kind}_{lm}_H_capture"] = float((h & ~accepted).sum() / n_h) if n_h else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_strict_clean_examples(test: pd.DataFrame, tau: float, outdir: Path) -> None:
    clean = test[test["perturbation"] == "none"].copy()
    clean["gate_accepted"] = accepted_mask(clean["ctc_support_nll"].astype(float), tau)
    clean["gate_caught"] = ~clean["gate_accepted"]
    strict = clean[clean["strict_h_union"]].copy()
    strict["lm_membership"] = np.where(
        strict["strict_h_both"],
        "both",
        np.where(strict["strict_h_qwen"], "qwen", "gpt2"),
    )
    strict = strict.sort_values(["strict_h_both", "WER"], ascending=[False, False])
    strict.to_csv(outdir / "strict_clean_hallucination_candidates.csv", index=False)


def render_headline(
    summary: pd.DataFrame,
    gate: pd.DataFrame,
    thresholds: Dict[str, float],
    tau: float,
) -> str:
    lines: List[str] = []
    lines.append("=== SeamlessM4T-v2 Large: acoustic-stress pipeline ===")
    lines.append(
        "Thresholds: diagnostic WER={:.4f}, strict WER={:.2f}, Qwen={:.4f}, GPT2={:.4f}, gate tau={:.6f}".format(
            thresholds["diagnostic_wer_threshold"],
            thresholds["strict_wer_threshold"],
            thresholds["qwen_plausibility_threshold"],
            thresholds["gpt2_plausibility_threshold"],
            tau,
        )
    )
    lines.append("")
    lines.append("TEST condition summary:")
    cols = [
        "condition",
        "WER_mean",
        "qwen_plaus_mean",
        "gpt2_plaus_mean",
        "diag_H_qwen_pct",
        "diag_H_gpt2_pct",
        "strict_H_qwen_pct",
        "strict_H_gpt2_pct",
        "rep34_pct",
        "top1_mass",
        "top10_mass",
    ]
    lines.append(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("Frozen gate TEST summary:")
    gcols = [
        "condition",
        "coverage",
        "WER_emitted",
        "strict_qwen_H_before",
        "strict_qwen_H_capture",
        "strict_gpt2_H_before",
        "strict_gpt2_H_capture",
    ]
    lines.append(gate[gcols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="SeamlessM4T-v2 Large acoustic-stress pipeline")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tgt_lang", default="eng")
    p.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--gpt2_model", default=DEFAULT_GPT2_MODEL)
    p.add_argument("--wav2vec2_model", default=DEFAULT_WAV2VEC2_MODEL)
    p.add_argument("--dev_tsv", type=Path, default=None)
    p.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    p.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    p.add_argument("--dev_max_samples", type=int, default=1000)
    p.add_argument("--test_max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lm_batch_size", type=int, default=8)
    p.add_argument("--ctc_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=225)
    p.add_argument("--strict_wer", type=float, default=DEFAULT_STRICT_WER)
    p.add_argument("--min_clean_coverage", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow_overlap", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a GPU SLURM allocation")
    device = "cuda"

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    generated_dir = outdir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    dev_tsv = resolve_dev_tsv(args.dev_tsv)
    dev_manifest = load_manifest(dev_tsv, args.clips_dir, split="dev", max_samples=args.dev_max_samples)
    test_manifest = load_manifest(args.test_tsv, args.clips_dir, split="test", max_samples=args.test_max_samples)
    validate_disjoint(dev_manifest, test_manifest, allow_overlap=args.allow_overlap)

    metadata = {
        "model": args.model,
        "target_language": args.tgt_lang,
        "pretrained_untouched": True,
        "dev_tsv": str(dev_tsv),
        "test_tsv": str(args.test_tsv),
        "clips_dir": str(args.clips_dir),
        "dev_max_samples": args.dev_max_samples,
        "test_max_samples": args.test_max_samples,
        "perturbations": DEFAULT_PERTURBATIONS,
        "strict_wer_threshold": args.strict_wer,
        "gate_min_clean_dev_coverage": args.min_clean_coverage,
        "seed": args.seed,
    }
    (outdir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    frames: List[pd.DataFrame] = []
    model = processor = None
    for split, manifest in (("dev", dev_manifest), ("test", test_manifest)):
        for condition in DEFAULT_PERTURBATIONS:
            cache = generated_dir / f"{split}_{condition}.csv"
            if args.resume and cache.exists():
                print(f"Reusing generated outputs: {cache}", flush=True)
                frame = pd.read_csv(cache)
            else:
                if model is None:
                    model, processor = load_seamless(args.model, device)
                print(f"Decoding {split} / {condition}", flush=True)
                frame = decode_manifest_seamless(
                    model,
                    processor,
                    manifest,
                    condition,
                    device=device,
                    batch_size=args.batch_size,
                    base_seed=args.seed,
                    tgt_lang=args.tgt_lang,
                    max_new_tokens=args.max_new_tokens,
                )
                frame.to_csv(cache, index=False)
            frames.append(frame)

    generated = pd.concat(frames, ignore_index=True)
    generated.to_csv(outdir / "generated_outputs.csv", index=False)
    if model is not None:
        del model, processor
        torch.cuda.empty_cache()

    lm_cache = outdir / "lm_wer_repetition_scored.csv"
    if args.resume and lm_cache.exists():
        print(f"Reusing LM/WER/repetition scores: {lm_cache}", flush=True)
        scored = pd.read_csv(lm_cache)
    else:
        scored = add_clean_wer(generated)
        scored = add_repetition(scored)
        scored = add_dual_lm_plausibility(
            scored,
            device=device,
            qwen_model=args.qwen_model,
            gpt2_model=args.gpt2_model,
            batch_size=args.lm_batch_size,
        )
        thresholds = derive_thresholds(scored, args.strict_wer)
        scored = apply_labels(scored, thresholds)
        scored.to_csv(lm_cache, index=False)

    thresholds = derive_thresholds(scored, args.strict_wer)
    scored = apply_labels(scored, thresholds)
    (outdir / "frozen_hallucination_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n"
    )

    ctc_cache = outdir / "wav2vec2_ctc_support_cache.jsonl"
    full_scored_cache = outdir / "scored_outputs.csv"
    if args.resume and full_scored_cache.exists():
        print(f"Reusing complete scored outputs: {full_scored_cache}", flush=True)
        scored = pd.read_csv(full_scored_cache)
        scored = apply_labels(scored, thresholds)
    else:
        scored = score_ctc_support(
            scored,
            wav2vec2_model=args.wav2vec2_model,
            device=device,
            cache_path=ctc_cache,
            batch_size=args.ctc_batch_size,
            base_seed=args.seed,
        )
        scored.to_csv(full_scored_cache, index=False)

    dev = scored[scored["split"] == "dev"].copy()
    test = scored[scored["split"] == "test"].copy()

    gate_dev = dev.copy()
    gate_dev["hallucination_like_qwen"] = gate_dev["strict_h_qwen"].astype(bool)
    gate_dev["hallucination_like_gpt2"] = gate_dev["strict_h_gpt2"].astype(bool)
    tau, calibration = select_gate_threshold(
        gate_dev,
        min_clean_coverage=args.min_clean_coverage,
    )
    calibration.to_csv(outdir / "gate_calibration_table.csv", index=False)
    gate_info = {
        "threshold": float(tau),
        "selection_split": "dev",
        "target_label": "strict_h_qwen",
        "strict_wer_threshold": float(args.strict_wer),
        "min_clean_dev_coverage": float(args.min_clean_coverage),
        "stress_conditions": [x for x in DEFAULT_PERTURBATIONS if x != "none"],
        "reference_free_at_test": True,
    }
    (outdir / "frozen_gate_threshold.json").write_text(
        json.dumps(gate_info, indent=2, sort_keys=True) + "\n"
    )

    dev_summary = summarize_conditions(scored, "dev")
    test_summary = summarize_conditions(scored, "test")
    dev_summary.to_csv(outdir / "dev_condition_summary.csv", index=False)
    test_summary.to_csv(outdir / "test_condition_summary.csv", index=False)

    gate_summary = gate_test_summary(test, tau)
    gate_summary.to_csv(outdir / "test_gate_summary.csv", index=False)
    save_strict_clean_examples(test, tau, outdir)

    headline = render_headline(test_summary, gate_summary, thresholds, tau)
    (outdir / "headline_summary.txt").write_text(headline)
    print("\n" + headline, flush=True)

    clean_test = test[test["perturbation"] == "none"].copy()
    strict_union = clean_test["strict_h_union"].astype(bool)
    gate_accept = accepted_mask(clean_test["ctc_support_nll"].astype(float), tau)
    report = {
        "model": args.model,
        "pretrained_untouched": True,
        "thresholds": thresholds,
        "gate": gate_info,
        "clean_test": {
            "N": int(len(clean_test)),
            "N_valid_reference": int(clean_test["valid_reference_cleanwer"].astype(bool).sum()),
            "WER_mean": float(clean_test["WER"].mean(skipna=True)),
            "WER_median": float(clean_test["WER"].median(skipna=True)),
            "strict_qwen_candidates": int(clean_test["strict_h_qwen"].astype(bool).sum()),
            "strict_gpt2_candidates": int(clean_test["strict_h_gpt2"].astype(bool).sum()),
            "strict_union_candidates": int(strict_union.sum()),
            "strict_union_gate_caught": int((strict_union & ~gate_accept).sum()),
            "heldout_clean_coverage": float(gate_accept.mean()),
        },
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
