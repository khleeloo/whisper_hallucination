#!/usr/bin/env python3
"""Canonical Base acoustic-stress experiment for the ASR hallucination pipeline.

Runs the matched Whisper Base checkpoint under the complete paper-facing acoustic
stress battery and scores every hypothesis with:
  * WER
  * Qwen3-0.6B normalized plausibility
  * GPT-2 normalized plausibility
  * 2/3/4-gram repetition
  * cross-utterance exact-output concentration

Hallucination-like labels are defined independently for Qwen3 and GPT-2 using
thresholds frozen from the matched clean Base rows in this run. The script keeps
hallucination and repetition separate and explicitly reports their overlap, so
severe high-WER/high-plausibility conditions cannot be mistaken for repetition
loops or stock-sentence collapse.

The full stress battery is always reported in a fixed, predeclared order,
including null/weak effects. This is intentional: the experiment tests which
forms of acoustic evidence loss induce hallucination-like behavior, not merely
whether any perturbation can increase it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from acoustic_abstention_mitigation import (
    DEFAULT_BASE_MODEL_DIR,
    DEFAULT_CLIPS_DIR,
    DEFAULT_TEST_TSV,
    add_dual_lm_scores,
    decode_manifest,
    load_manifest,
    stable_seed,
)
from evaluate_whisper_validation import compute_repetition_metrics, normalize_text
from mitigation_experiment import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GPT2_MODEL,
    DEFAULT_QWEN_MODEL,
    _load_whisper_model,
)

SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_OUTPUT_DIR = SCRATCH_ROOT / "acoustic_stress_full"

# Fixed order for the main paper table. Do not sort by effect size.
PERTURBATION_ORDER = [
    "none",
    "silence_amp0.0_dur0.0",
    "leading_silence_amp0.0_dur1.0",
    "leading_silence_amp0.0_dur3.0",
    "onset_noise_amp0.05_dur0.5",
    "onset_noise_amp0.5_dur0.5",
    "onset_noise_amp0.75_dur0.5",
    "reverb_amp0.5_dur0.5",
    "reverb_amp0.8_dur0.5",
    "speech_band_noise_amp0.5_dur0.0",
    "speech_band_noise_amp0.75_dur0.0",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]


def perturbation_family(label: str) -> str:
    if label == "none":
        return "none"
    for family in [
        "leading_silence",
        "speech_band_noise",
        "onset_noise",
        "full_noise",
        "reverb",
        "silence",
    ]:
        if label.startswith(family):
            return family
    return "other"


def bootstrap_mean_ci(
    values: Sequence[float], *, n_boot: int, seed: int
) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        boot[i] = x[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def add_repetition_and_collapse_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rep_rows = [compute_repetition_metrics(text) for text in out["hypothesis"].astype(str)]
    out["rep2"] = [int(row["bigram_rep_count"]) for row in rep_rows]
    out["rep3"] = [int(row["trigram_rep_count"]) for row in rep_rows]
    out["rep4"] = [int(row["fourgram_rep_count"]) for row in rep_rows]
    out["repetition_3_or_4"] = (out["rep3"] > 0) | (out["rep4"] > 0)

    out["hypothesis_norm"] = out["hypothesis"].fillna("").astype(str).map(normalize_text)
    out["reference_norm"] = out["reference"].fillna("").astype(str).map(normalize_text)
    out["hyp_words"] = out["hypothesis_norm"].map(lambda x: len(x.split()))
    out["ref_words"] = out["reference_norm"].map(lambda x: len(x.split()))
    out["empty_output"] = out["hyp_words"].eq(0)
    out["length_ratio"] = np.where(
        out["ref_words"].gt(0),
        out["hyp_words"] / out["ref_words"],
        np.nan,
    )

    # Exact-output concentration is computed within each acoustic condition.
    # Empty outputs are handled separately and are not treated as stock phrases.
    frequencies = pd.Series(0, index=out.index, dtype=int)
    duplicate = pd.Series(False, index=out.index, dtype=bool)
    for _, indices in out.groupby("perturbation", sort=False).groups.items():
        idx = list(indices)
        texts = out.loc[idx, "hypothesis_norm"]
        counts = texts[texts.ne("")].value_counts()
        for row_idx, text in zip(idx, texts):
            if text:
                count = int(counts.get(text, 0))
                frequencies.loc[row_idx] = count
                duplicate.loc[row_idx] = count > 1
    out["same_hypothesis_frequency"] = frequencies.to_numpy()
    out["cross_utterance_duplicate"] = duplicate.to_numpy()
    return out


def derive_matched_clean_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    clean = df[df["perturbation"].eq("none")].copy()
    if clean.empty:
        raise ValueError("Matched clean Base rows are required to freeze thresholds")
    thresholds = {
        "wer_threshold": float(pd.to_numeric(clean["WER"], errors="coerce").mean()),
        "qwen_plausibility_threshold": float(
            pd.to_numeric(clean["qwen_plaus"], errors="coerce").mean()
        ),
        "gpt2_plausibility_threshold": float(
            pd.to_numeric(clean["gpt2_plaus"], errors="coerce").mean()
        ),
        "N_clean": int(len(clean)),
    }
    if not all(np.isfinite(float(v)) for k, v in thresholds.items() if k != "N_clean"):
        raise ValueError(f"Non-finite matched clean thresholds: {thresholds}")
    return thresholds


def apply_dual_hallucination_labels(
    df: pd.DataFrame, thresholds: Dict[str, float]
) -> pd.DataFrame:
    out = df.copy()
    high_wer = out["WER"].astype(float) > float(thresholds["wer_threshold"])
    out["hallucination_qwen"] = high_wer & (
        out["qwen_plaus"].astype(float)
        > float(thresholds["qwen_plausibility_threshold"])
    )
    out["hallucination_gpt2"] = high_wer & (
        out["gpt2_plaus"].astype(float)
        > float(thresholds["gpt2_plausibility_threshold"])
    )
    out["hallucination_lm_agreement"] = (
        out["hallucination_qwen"] == out["hallucination_gpt2"]
    )

    rep = out["repetition_3_or_4"].astype(bool)
    dup = out["cross_utterance_duplicate"].astype(bool)
    nonempty = ~out["empty_output"].astype(bool)
    for lm in ["qwen", "gpt2"]:
        hall = out[f"hallucination_{lm}"].astype(bool)
        out[f"{lm}_hall_only_no_rep"] = hall & ~rep
        out[f"{lm}_hall_plus_rep"] = hall & rep
        out[f"{lm}_hall_nonrep_unique"] = hall & ~rep & ~dup & nonempty
        out[f"{lm}_hall_duplicate_output"] = hall & dup
    return out


def pair_to_clean(df: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["perturbation"].eq("none")][
        [
            "utterance_id",
            "WER",
            "hallucination_qwen",
            "hallucination_gpt2",
            "repetition_3_or_4",
        ]
    ].copy()
    clean = clean.rename(
        columns={
            "WER": "WER_clean",
            "hallucination_qwen": "hallucination_qwen_clean",
            "hallucination_gpt2": "hallucination_gpt2_clean",
            "repetition_3_or_4": "repetition_3_or_4_clean",
        }
    )
    if clean["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance IDs in clean Base rows")
    paired = group.merge(clean, on="utterance_id", how="left", validate="one_to_one", sort=False)
    if paired["WER_clean"].isna().any():
        raise ValueError(
            f"Could not pair {int(paired['WER_clean'].isna().sum())} rows to clean Base"
        )
    return paired


def concentration_stats(group: pd.DataFrame) -> Dict[str, object]:
    nonempty = group.loc[~group["empty_output"].astype(bool), "hypothesis_norm"]
    if nonempty.empty:
        return {
            "unique_hypothesis_fraction_nonempty": float("nan"),
            "top1_hypothesis_mass_nonempty": float("nan"),
            "top10_hypothesis_mass_nonempty": float("nan"),
            "most_common_hypothesis": "",
            "most_common_hypothesis_count": 0,
        }
    counts = nonempty.value_counts()
    return {
        "unique_hypothesis_fraction_nonempty": float(len(counts) / len(nonempty)),
        "top1_hypothesis_mass_nonempty": float(counts.iloc[0] / len(nonempty)),
        "top10_hypothesis_mass_nonempty": float(counts.iloc[:10].sum() / len(nonempty)),
        "most_common_hypothesis": str(counts.index[0]),
        "most_common_hypothesis_count": int(counts.iloc[0]),
    }


def summarize_condition(
    all_rows: pd.DataFrame,
    group: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    perturbation = str(group.iloc[0]["perturbation"])
    paired = pair_to_clean(all_rows, group)

    delta_wer = paired["WER"].astype(float).to_numpy() - paired["WER_clean"].astype(float).to_numpy()
    q_delta = (
        paired["hallucination_qwen"].astype(float).to_numpy()
        - paired["hallucination_qwen_clean"].astype(float).to_numpy()
    )
    g_delta = (
        paired["hallucination_gpt2"].astype(float).to_numpy()
        - paired["hallucination_gpt2_clean"].astype(float).to_numpy()
    )
    r_delta = (
        paired["repetition_3_or_4"].astype(float).to_numpy()
        - paired["repetition_3_or_4_clean"].astype(float).to_numpy()
    )

    wer_lo, wer_hi = bootstrap_mean_ci(
        delta_wer, n_boot=n_boot, seed=stable_seed(seed, perturbation, "WER")
    )
    q_lo, q_hi = bootstrap_mean_ci(
        q_delta, n_boot=n_boot, seed=stable_seed(seed, perturbation, "Qwen")
    )
    g_lo, g_hi = bootstrap_mean_ci(
        g_delta, n_boot=n_boot, seed=stable_seed(seed, perturbation, "GPT2")
    )
    r_lo, r_hi = bootstrap_mean_ci(
        r_delta, n_boot=n_boot, seed=stable_seed(seed, perturbation, "Rep34")
    )

    q_hall = group["hallucination_qwen"].astype(bool)
    g_hall = group["hallucination_gpt2"].astype(bool)
    rep = group["repetition_3_or_4"].astype(bool)

    row: Dict[str, object] = {
        "perturbation": perturbation,
        "family": perturbation_family(perturbation),
        "N": int(len(group)),
        "mean_WER": float(group["WER"].astype(float).mean()),
        "delta_WER": float(np.mean(delta_wer)),
        "delta_WER_CI_low": wer_lo,
        "delta_WER_CI_high": wer_hi,
        "mean_qwen_plaus": float(group["qwen_plaus"].astype(float).mean()),
        "mean_gpt2_plaus": float(group["gpt2_plaus"].astype(float).mean()),
        "hallucination_qwen_rate": float(q_hall.mean()),
        "delta_hallucination_qwen": float(np.mean(q_delta)),
        "delta_hallucination_qwen_CI_low": q_lo,
        "delta_hallucination_qwen_CI_high": q_hi,
        "hallucination_gpt2_rate": float(g_hall.mean()),
        "delta_hallucination_gpt2": float(np.mean(g_delta)),
        "delta_hallucination_gpt2_CI_low": g_lo,
        "delta_hallucination_gpt2_CI_high": g_hi,
        "lm_hallucination_agreement_rate": float(
            group["hallucination_lm_agreement"].astype(bool).mean()
        ),
        "rep2_rate": float((group["rep2"] > 0).mean()),
        "rep3_rate": float((group["rep3"] > 0).mean()),
        "rep4_rate": float((group["rep4"] > 0).mean()),
        "rep3_or_rep4_rate": float(rep.mean()),
        "delta_rep3_or_rep4": float(np.mean(r_delta)),
        "delta_rep3_or_rep4_CI_low": r_lo,
        "delta_rep3_or_rep4_CI_high": r_hi,
        "qwen_hall_only_no_rep_rate": float(
            group["qwen_hall_only_no_rep"].astype(bool).mean()
        ),
        "qwen_hall_plus_rep_rate": float(
            group["qwen_hall_plus_rep"].astype(bool).mean()
        ),
        "qwen_fraction_hallucinations_with_rep": (
            float((q_hall & rep).sum() / q_hall.sum()) if int(q_hall.sum()) else 0.0
        ),
        "qwen_hall_nonrep_unique_rate": float(
            group["qwen_hall_nonrep_unique"].astype(bool).mean()
        ),
        "qwen_hall_duplicate_output_rate": float(
            group["qwen_hall_duplicate_output"].astype(bool).mean()
        ),
        "gpt2_hall_only_no_rep_rate": float(
            group["gpt2_hall_only_no_rep"].astype(bool).mean()
        ),
        "gpt2_hall_plus_rep_rate": float(
            group["gpt2_hall_plus_rep"].astype(bool).mean()
        ),
        "gpt2_fraction_hallucinations_with_rep": (
            float((g_hall & rep).sum() / g_hall.sum()) if int(g_hall.sum()) else 0.0
        ),
        "gpt2_hall_nonrep_unique_rate": float(
            group["gpt2_hall_nonrep_unique"].astype(bool).mean()
        ),
        "gpt2_hall_duplicate_output_rate": float(
            group["gpt2_hall_duplicate_output"].astype(bool).mean()
        ),
        "empty_output_rate": float(group["empty_output"].astype(bool).mean()),
        "mean_hyp_words": float(group["hyp_words"].astype(float).mean()),
        "mean_ref_words": float(group["ref_words"].astype(float).mean()),
        "mean_length_ratio": float(
            pd.to_numeric(group["length_ratio"], errors="coerce").mean()
        ),
        "median_length_ratio": float(
            pd.to_numeric(group["length_ratio"], errors="coerce").median()
        ),
    }
    row.update(concentration_stats(group))
    return row


def build_top_hypotheses(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    order = {label: i for i, label in enumerate(PERTURBATION_ORDER)}
    for perturbation, group in df.groupby("perturbation", sort=False):
        nonempty = group[~group["empty_output"].astype(bool)].copy()
        if nonempty.empty:
            continue
        counts = Counter(nonempty["hypothesis_norm"].tolist())
        for rank, (text, count) in enumerate(counts.most_common(top_k), start=1):
            subset = nonempty[nonempty["hypothesis_norm"].eq(text)]
            rows.append(
                {
                    "perturbation": perturbation,
                    "order": order.get(perturbation, 999),
                    "rank": rank,
                    "hypothesis": text,
                    "count": int(count),
                    "mass_nonempty": float(count / len(nonempty)),
                    "qwen_hallucination_rate": float(
                        subset["hallucination_qwen"].astype(bool).mean()
                    ),
                    "gpt2_hallucination_rate": float(
                        subset["hallucination_gpt2"].astype(bool).mean()
                    ),
                    "rep3_or_rep4_rate": float(
                        subset["repetition_3_or_4"].astype(bool).mean()
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "perturbation",
                "rank",
                "hypothesis",
                "count",
                "mass_nonempty",
                "qwen_hallucination_rate",
                "gpt2_hallucination_rate",
                "rep3_or_rep4_rate",
            ]
        )
    out = pd.DataFrame(rows).sort_values(["order", "rank"]).drop(columns=["order"])
    return out.reset_index(drop=True)


def generate_outputs(
    manifest: pd.DataFrame,
    *,
    model_dir: Path,
    base_model_name: str,
    perturbations: Sequence[str],
    device: str,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    model, base_model, processor = _load_whisper_model(
        model_dir, base_model_name, device
    )
    frames: List[pd.DataFrame] = []
    try:
        for perturbation in perturbations:
            print(f"\n=== Decoding {perturbation} ===", flush=True)
            frames.append(
                decode_manifest(
                    model,
                    processor,
                    manifest,
                    perturbation,
                    device=device,
                    batch_size=batch_size,
                    base_seed=seed,
                )
            )
    finally:
        model.cpu()
        del model, base_model, processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def validate_complete_battery(df: pd.DataFrame, perturbations: Sequence[str], n_expected: int) -> None:
    present = set(df["perturbation"].astype(str))
    missing = [label for label in perturbations if label not in present]
    if missing:
        raise ValueError(f"Missing acoustic conditions: {missing}")
    extras = sorted(present - set(perturbations))
    if extras:
        raise ValueError(f"Unexpected acoustic conditions in generated outputs: {extras}")
    sizes = df.groupby("perturbation").size().to_dict()
    bad = {
        label: int(sizes.get(label, 0))
        for label in perturbations
        if int(sizes.get(label, 0)) != n_expected
    }
    if bad:
        raise ValueError(
            f"Every acoustic condition must contain the same {n_expected} matched utterances; got {bad}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--base_model_dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--gpt2_model", default=DEFAULT_GPT2_MODEL)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lm_batch_size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse_generated_outputs", action="store_true")
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=PERTURBATION_ORDER,
        help="Override only for debugging. Paper runs should use the default full battery.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.output_dir / "generated_outputs.csv"
    scored_path = args.output_dir / "per_utterance_acoustic_stress.csv"
    summary_path = args.output_dir / "acoustic_stress_summary.csv"
    top_path = args.output_dir / "top_hypotheses_by_condition.csv"
    thresholds_path = args.output_dir / "matched_clean_dual_lm_thresholds.json"
    report_path = args.output_dir / "report.json"

    print("=== Canonical Base acoustic-stress experiment ===", flush=True)
    print(f"Model: {args.base_model_dir}", flush=True)
    print(f"Test TSV: {args.test_tsv}", flush=True)
    print(f"Max samples: {args.max_samples}", flush=True)
    print(f"Qwen: {args.qwen_model}", flush=True)
    print(f"GPT2: {args.gpt2_model}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print("Conditions (fixed order):", flush=True)
    for label in args.perturbations:
        print(f"  - {label}", flush=True)

    manifest = load_manifest(
        args.test_tsv,
        args.clips_dir,
        split="test",
        max_samples=args.max_samples,
    )
    print(f"Matched utterances: {len(manifest):,}", flush=True)

    if args.reuse_generated_outputs and generated_path.exists():
        generated = pd.read_csv(generated_path)
        print(f"Reusing generated hypotheses: {generated_path}", flush=True)
    else:
        generated = generate_outputs(
            manifest,
            model_dir=args.base_model_dir,
            base_model_name=args.base_model_name,
            perturbations=args.perturbations,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        generated.to_csv(generated_path, index=False)
        print(f"Saved generated hypotheses: {generated_path}", flush=True)

    validate_complete_battery(generated, args.perturbations, len(manifest))

    print("\n=== Dual-LM scoring ===", flush=True)
    scored = add_dual_lm_scores(
        generated,
        device=device,
        qwen_model=args.qwen_model,
        gpt2_model=args.gpt2_model,
        lm_batch_size=args.lm_batch_size,
    )
    scored = add_repetition_and_collapse_features(scored)
    thresholds = derive_matched_clean_thresholds(scored)
    scored = apply_dual_hallucination_labels(scored, thresholds)
    scored.to_csv(scored_path, index=False)

    thresholds_payload = {
        **thresholds,
        "criterion": "WER > matched clean Base mean AND LM plausibility > matched clean Base mean",
        "qwen_model": args.qwen_model,
        "gpt2_model": args.gpt2_model,
        "thresholds_frozen_before_stress_comparison": True,
        "note": (
            "Qwen3 and GPT2 hallucination-like labels are separate. "
            "Repetition is never folded into the hallucination definition."
        ),
    }
    thresholds_path.write_text(
        json.dumps(thresholds_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    order = {label: i for i, label in enumerate(args.perturbations)}
    summary_rows = []
    for perturbation in args.perturbations:
        group = scored[scored["perturbation"].eq(perturbation)].copy()
        summary_rows.append(
            summarize_condition(
                scored,
                group,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary["order"] = summary["perturbation"].map(order)
    summary = summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    top = build_top_hypotheses(scored, top_k=10)
    top.to_csv(top_path, index=False)

    core_cols = [
        "perturbation",
        "N",
        "mean_WER",
        "mean_qwen_plaus",
        "mean_gpt2_plaus",
        "hallucination_qwen_rate",
        "delta_hallucination_qwen",
        "hallucination_gpt2_rate",
        "delta_hallucination_gpt2",
        "rep3_or_rep4_rate",
        "qwen_hall_only_no_rep_rate",
        "qwen_hall_plus_rep_rate",
        "qwen_hall_nonrep_unique_rate",
        "empty_output_rate",
        "unique_hypothesis_fraction_nonempty",
        "top1_hypothesis_mass_nonempty",
        "top10_hypothesis_mass_nonempty",
    ]
    print("\n=== Full acoustic stress comparison ===", flush=True)
    print(
        summary[core_cols].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        flush=True,
    )

    report = {
        "experiment": "canonical_base_acoustic_stress",
        "purpose": (
            "Test which forms of acoustic evidence loss produce hallucination-like "
            "behavior while separating hallucination from repetition and exact-output collapse."
        ),
        "model_dir": str(args.base_model_dir),
        "test_tsv": str(args.test_tsv),
        "N_per_condition": int(len(manifest)),
        "perturbations": list(args.perturbations),
        "qwen_model": args.qwen_model,
        "gpt2_model": args.gpt2_model,
        "thresholds": thresholds_payload,
        "repetition_definition": "rep3 > 0 OR rep4 > 0",
        "collapse_definition": (
            "Exact normalized hypothesis frequency within an acoustic condition; "
            "empty outputs are reported separately and excluded from concentration statistics."
        ),
        "outputs": {
            "per_utterance": str(scored_path),
            "summary": str(summary_path),
            "top_hypotheses": str(top_path),
            "thresholds": str(thresholds_path),
        },
        "seed": args.seed,
        "bootstrap": args.bootstrap,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs:", flush=True)
    for path in [scored_path, summary_path, top_path, thresholds_path, report_path]:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
