#!/usr/bin/env python3
"""Test whether failure diagnosis predicts which mitigation is useful.

This experiment compares two simple, deliberately different interventions:

1. Collapse-aware abstention.  Build a small lexicon from the most frequent
   normalized outputs on stressed DEV, while preserving a requested fraction of
   clean DEV.  At TEST time, exact matches to this frozen lexicon are rejected.
   This targets cross-utterance decoder-default / mode-collapse behavior.

2. Anti-repetition decoding.  Re-decode held-out TEST speech with a fixed
   repetition penalty.  This targets repetition-oriented degeneration without
   using TEST labels or references to tune the intervention.

The point is not to propose either intervention as a universal ASR solution.
The experiment asks whether the multi-axis diagnostic phenotype has decision
utility: a collapse-heavy model should benefit more from collapse-aware
abstention, whereas a repetition-heavy model should show larger gains from an
anti-repetition decoding intervention.

All evaluation uses the corrected WER normalizer and the same frozen clean-DEV
Qwen3/GPT-2 plausibility thresholds as the baseline model.  Conditions are the
matched protocol used throughout the paper: clean, full-noise 0.50, full-noise
0.75.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from acoustic_abstention_mitigation import load_exact_waveform
from clean_wer_rescore import add_clean_wer, normalize_asr_text
from evaluate_whisper_validation import compute_repetition_metrics
from mitigation_experiment import DEFAULT_GPT2_MODEL, DEFAULT_QWEN_MODEL, _default_lm_plausibility

CONDITIONS = [
    "none",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]
STRICT_WER = 0.5


def add_repetition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metrics = [compute_repetition_metrics(str(h)) for h in out["hypothesis"]]
    out["rep2"] = [int(m.get("bigram_rep_count", 0)) for m in metrics]
    out["rep3"] = [int(m.get("trigram_rep_count", 0)) for m in metrics]
    out["rep4"] = [int(m.get("fourgram_rep_count", 0)) for m in metrics]
    out["rep34"] = (out["rep3"] > 0) | (out["rep4"] > 0)
    return out


def add_lm_scores(
    df: pd.DataFrame,
    *,
    device: str,
    qwen_model: str,
    gpt2_model: str,
    batch_size: int,
) -> pd.DataFrame:
    out = df.copy()
    for column, model_name in (("qwen_plaus", qwen_model), ("gpt2_plaus", gpt2_model)):
        print(f"Scoring {len(out):,} outputs with {model_name}...", flush=True)
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


def derive_thresholds(baseline: pd.DataFrame) -> Dict[str, float]:
    clean = baseline[
        (baseline["split"].astype(str) == "dev")
        & (baseline["perturbation"].astype(str) == "none")
        & baseline["valid_reference_cleanwer"].astype(bool)
        & np.isfinite(pd.to_numeric(baseline["WER"], errors="coerce"))
    ].copy()
    if clean.empty:
        raise ValueError("No valid clean DEV rows for threshold derivation")
    return {
        "diagnostic_wer_threshold": float(clean["WER"].astype(float).mean()),
        "strict_wer_threshold": STRICT_WER,
        "qwen_plausibility_threshold": float(clean["qwen_plaus"].astype(float).mean()),
        "gpt2_plausibility_threshold": float(clean["gpt2_plaus"].astype(float).mean()),
        "N_clean_dev_valid": int(len(clean)),
    }


def apply_labels(df: pd.DataFrame, th: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    valid = out["valid_reference_cleanwer"].astype(bool) & np.isfinite(out["WER"].astype(float))
    diag_wer = valid & (out["WER"].astype(float) > th["diagnostic_wer_threshold"])
    strict_wer = valid & (out["WER"].astype(float) > th["strict_wer_threshold"])
    for lm in ("qwen", "gpt2"):
        high_plaus = out[f"{lm}_plaus"].astype(float) > th[f"{lm}_plausibility_threshold"]
        out[f"diag_h_{lm}"] = diag_wer & high_plaus
        out[f"strict_h_{lm}"] = strict_wer & high_plaus
    out["strict_h_union"] = out["strict_h_qwen"] | out["strict_h_gpt2"]
    out["strict_h_both"] = out["strict_h_qwen"] & out["strict_h_gpt2"]
    return out


def concentration(hypotheses: Sequence[object]) -> Tuple[float, float]:
    norm = [normalize_asr_text(x) for x in hypotheses]
    norm = [x for x in norm if x]
    if not norm:
        return 0.0, 0.0
    counts = pd.Series(norm).value_counts()
    return float(counts.iloc[0] / len(norm)), float(counts.iloc[:10].sum() / len(norm))


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        g = df[df["perturbation"].astype(str) == condition].copy()
        if g.empty:
            continue
        valid = g["valid_reference_cleanwer"].astype(bool)
        gv = g[valid]
        top1, top10 = concentration(g["hypothesis"].tolist())
        rows.append({
            "variant": label,
            "condition": condition,
            "N": int(len(g)),
            "WER": float(gv["WER"].mean()) if len(gv) else float("nan"),
            "qwen_plaus": float(g["qwen_plaus"].astype(float).mean()),
            "gpt2_plaus": float(g["gpt2_plaus"].astype(float).mean()),
            "diag_H_qwen_pct": 100.0 * float(g["diag_h_qwen"].astype(bool).mean()),
            "diag_H_gpt2_pct": 100.0 * float(g["diag_h_gpt2"].astype(bool).mean()),
            "strict_H_qwen_pct": 100.0 * float(g["strict_h_qwen"].astype(bool).mean()),
            "strict_H_gpt2_pct": 100.0 * float(g["strict_h_gpt2"].astype(bool).mean()),
            "strict_H_union_pct": 100.0 * float(g["strict_h_union"].astype(bool).mean()),
            "rep34_pct": 100.0 * float(g["rep34"].astype(bool).mean()),
            "top1_mass": top1,
            "top10_mass": top10,
        })
    return pd.DataFrame(rows)


def calibrate_collapse_lexicon(
    baseline: pd.DataFrame,
    *,
    min_clean_coverage: float,
    max_entries: int,
) -> pd.DataFrame:
    """Freeze a compact stress-output lexicon using DEV only.

    Candidates are ranked by pooled severe-stress DEV frequency.  We add an
    output only when doing so keeps clean-DEV coverage above the requested
    minimum.  No TEST information is used.
    """
    dev = baseline[baseline["split"].astype(str) == "dev"].copy()
    dev["hyp_norm"] = dev["hypothesis"].map(normalize_asr_text)
    stress = dev[dev["perturbation"].astype(str) != "none"].copy()
    clean = dev[dev["perturbation"].astype(str) == "none"].copy()

    stress = stress[stress["hyp_norm"] != ""]
    counts = stress["hyp_norm"].value_counts()
    clean_counts = clean["hyp_norm"].value_counts()
    selected: List[Dict[str, object]] = []
    lexicon: set[str] = set()

    for hyp, count in counts.items():
        if len(selected) >= max_entries:
            break
        trial = set(lexicon)
        trial.add(str(hyp))
        clean_reject = float(clean["hyp_norm"].isin(trial).mean())
        clean_coverage = 1.0 - clean_reject
        if clean_coverage + 1e-12 < min_clean_coverage:
            continue
        lexicon = trial
        selected.append({
            "rank": len(selected) + 1,
            "normalized_hypothesis": str(hyp),
            "stress_dev_count": int(count),
            "stress_dev_mass": float(count / len(stress)),
            "clean_dev_count": int(clean_counts.get(hyp, 0)),
            "clean_dev_mass": float(clean_counts.get(hyp, 0) / len(clean)) if len(clean) else 0.0,
            "clean_dev_coverage_after_add": clean_coverage,
        })

    return pd.DataFrame(selected)


def collapse_rejector_summary(
    baseline_test: pd.DataFrame,
    lexicon_df: pd.DataFrame,
) -> pd.DataFrame:
    lexicon = set(lexicon_df["normalized_hypothesis"].astype(str)) if len(lexicon_df) else set()
    test = baseline_test.copy()
    test["hyp_norm"] = test["hypothesis"].map(normalize_asr_text)
    test["collapse_rejected"] = test["hyp_norm"].isin(lexicon)
    rows: List[Dict[str, object]] = []

    for condition in CONDITIONS:
        g = test[test["perturbation"].astype(str) == condition].copy()
        if g.empty:
            continue
        accepted = ~g["collapse_rejected"].astype(bool)
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
                row[f"{kind}_{lm}_capture"] = float((h & ~accepted.to_numpy()).sum() / n_h) if n_h else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_test_manifest(baseline: pd.DataFrame) -> pd.DataFrame:
    clean = baseline[
        (baseline["split"].astype(str) == "test")
        & (baseline["perturbation"].astype(str) == "none")
    ].copy()
    cols = ["split", "utterance_id", "audio_path", "reference"]
    clean = clean[cols].drop_duplicates("utterance_id").reset_index(drop=True)
    if clean.empty:
        raise ValueError("No clean TEST manifest rows found in baseline source")
    return clean


def decode_whisper_antirep(
    model: object,
    processor: object,
    manifest: pd.DataFrame,
    perturbation: str,
    *,
    device: str,
    batch_size: int,
    base_seed: int,
    repetition_penalty: float,
) -> pd.DataFrame:
    hypotheses: List[str] = []
    records = list(manifest.itertuples(index=False))
    model_dtype = next(model.parameters()).dtype
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        waveforms = [load_exact_waveform(r, perturbation, base_seed=base_seed) for r in batch]
        inputs = processor.feature_extractor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(device)
        inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)
        with torch.inference_mode():
            ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=225,
                language="en",
                task="transcribe",
                repetition_penalty=repetition_penalty,
            )
        hypotheses.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        done = min(start + len(batch), len(records))
        if start == 0 or done == len(records) or (start // batch_size) % 10 == 0:
            print(f"  Whisper anti-rep {perturbation}: {done}/{len(records)}", flush=True)
    out = manifest[["split", "utterance_id", "audio_path", "reference"]].copy()
    out["perturbation"] = perturbation
    out["hypothesis"] = hypotheses
    return out


def decode_seamless_antirep(
    model: object,
    processor: object,
    manifest: pd.DataFrame,
    perturbation: str,
    *,
    device: str,
    batch_size: int,
    base_seed: int,
    repetition_penalty: float,
    tgt_lang: str,
) -> pd.DataFrame:
    hypotheses: List[str] = []
    records = list(manifest.itertuples(index=False))
    model_dtype = next(model.parameters()).dtype
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        waveforms = [load_exact_waveform(r, perturbation, base_seed=base_seed) for r in batch]
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
            ids = model.generate(
                **moved,
                tgt_lang=tgt_lang,
                max_new_tokens=225,
                repetition_penalty=repetition_penalty,
            )
        hypotheses.extend([str(x).strip() for x in processor.batch_decode(ids, skip_special_tokens=True)])
        done = min(start + len(batch), len(records))
        if start == 0 or done == len(records) or (start // batch_size) % 10 == 0:
            print(f"  Seamless anti-rep {perturbation}: {done}/{len(records)}", flush=True)
    out = manifest[["split", "utterance_id", "audio_path", "reference"]].copy()
    out["perturbation"] = perturbation
    out["hypothesis"] = hypotheses
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phenotype-targeted mitigation decision-utility experiment")
    p.add_argument("--model_type", choices=["raw_whisper", "seamless"], required=True)
    p.add_argument("--baseline_source", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--whisper_model", default="openai/whisper-large-v3")
    p.add_argument("--seamless_model", default="facebook/seamless-m4t-v2-large")
    p.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--gpt2_model", default=DEFAULT_GPT2_MODEL)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lm_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--collapse_min_clean_coverage", type=float, default=0.99)
    p.add_argument("--collapse_max_entries", type=int, default=20)
    p.add_argument("--tgt_lang", default="eng")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = "cuda"
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline: {args.baseline_source}", flush=True)
    baseline = pd.read_csv(args.baseline_source)
    baseline = add_clean_wer(baseline)
    if not {"rep34", "rep3", "rep4"}.issubset(baseline.columns):
        baseline = add_repetition(baseline)
    thresholds = derive_thresholds(baseline)
    baseline = apply_labels(baseline, thresholds)
    (outdir / "frozen_evaluation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n"
    )

    baseline_test = baseline[baseline["split"].astype(str) == "test"].copy()
    baseline_summary = summarize(baseline_test, "baseline")
    baseline_summary.to_csv(outdir / "baseline_test_summary.csv", index=False)

    # 1) Collapse-aware abstention: DEV-only calibration, frozen TEST application.
    lexicon = calibrate_collapse_lexicon(
        baseline,
        min_clean_coverage=args.collapse_min_clean_coverage,
        max_entries=args.collapse_max_entries,
    )
    lexicon.to_csv(outdir / "collapse_lexicon_dev.csv", index=False)
    collapse_summary = collapse_rejector_summary(baseline_test, lexicon)
    collapse_summary.to_csv(outdir / "collapse_rejector_test_summary.csv", index=False)

    # 2) Fixed anti-repetition decoding on TEST only.  Hyperparameter is fixed
    # before TEST and is not selected from TEST performance.
    generated_path = outdir / "antirep_generated_test.csv"
    if args.resume and generated_path.exists():
        anti = pd.read_csv(generated_path)
    else:
        manifest = build_test_manifest(baseline)
        frames: List[pd.DataFrame] = []
        if args.model_type == "raw_whisper":
            from pretrained_whisper_stress_pipeline import load_pretrained_whisper
            model, processor = load_pretrained_whisper(args.whisper_model, device)
            for condition in CONDITIONS:
                frames.append(
                    decode_whisper_antirep(
                        model,
                        processor,
                        manifest,
                        condition,
                        device=device,
                        batch_size=args.batch_size,
                        base_seed=args.seed,
                        repetition_penalty=args.repetition_penalty,
                    )
                )
        else:
            from seamless_m4t_stress_pipeline import load_seamless
            model, processor = load_seamless(args.seamless_model, device)
            for condition in CONDITIONS:
                frames.append(
                    decode_seamless_antirep(
                        model,
                        processor,
                        manifest,
                        condition,
                        device=device,
                        batch_size=args.batch_size,
                        base_seed=args.seed,
                        repetition_penalty=args.repetition_penalty,
                        tgt_lang=args.tgt_lang,
                    )
                )
        anti = pd.concat(frames, ignore_index=True)
        anti.to_csv(generated_path, index=False)
        del model, processor
        torch.cuda.empty_cache()

    scored_path = outdir / "antirep_scored_test.csv"
    if args.resume and scored_path.exists():
        anti = pd.read_csv(scored_path)
        anti = apply_labels(anti, thresholds)
    else:
        anti = add_clean_wer(anti)
        anti = add_repetition(anti)
        anti = add_lm_scores(
            anti,
            device=device,
            qwen_model=args.qwen_model,
            gpt2_model=args.gpt2_model,
            batch_size=args.lm_batch_size,
        )
        anti = apply_labels(anti, thresholds)
        anti.to_csv(scored_path, index=False)

    anti_summary = summarize(anti, f"repetition_penalty_{args.repetition_penalty:g}")
    anti_summary.to_csv(outdir / "antirep_test_summary.csv", index=False)

    comparison = baseline_summary.merge(
        anti_summary,
        on="condition",
        suffixes=("_baseline", "_antirep"),
    )
    for metric in (
        "WER",
        "strict_H_qwen_pct",
        "strict_H_gpt2_pct",
        "rep34_pct",
        "top1_mass",
        "top10_mass",
    ):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_antirep"] - comparison[f"{metric}_baseline"]
    comparison.to_csv(outdir / "antirep_vs_baseline.csv", index=False)

    report = {
        "model_type": args.model_type,
        "baseline_source": str(args.baseline_source),
        "conditions": CONDITIONS,
        "seed": args.seed,
        "repetition_penalty": args.repetition_penalty,
        "collapse_min_clean_coverage": args.collapse_min_clean_coverage,
        "collapse_max_entries": args.collapse_max_entries,
        "collapse_lexicon_size": int(len(lexicon)),
        "thresholds": thresholds,
    }
    (outdir / "experiment_metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== Baseline ===")
    print(baseline_summary.to_string(index=False), flush=True)
    print("\n=== Collapse-aware abstention ===")
    print(collapse_summary.to_string(index=False), flush=True)
    print("\n=== Anti-repetition decoding ===")
    print(anti_summary.to_string(index=False), flush=True)
    print("\n=== Anti-repetition deltas ===")
    delta_cols = ["condition"] + [c for c in comparison.columns if c.startswith("delta_")]
    print(comparison[delta_cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
