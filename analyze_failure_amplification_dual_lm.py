#!/usr/bin/env python3
"""Dual-LM structured-mismatch / acoustic-stress analysis.

This is the paper-facing version of the failure-signature analysis. It reuses
pairing, bootstrap, DID, and signature functions from
``analyze_failure_amplification.py`` but defines hallucination-like outputs for
both language models used by the evaluation pipeline:

  HallucinationQwen = WER > mean(Base clean WER)
                      AND Qwen > mean(Base clean Qwen)
  HallucinationGPT2 = WER > mean(Base clean WER)
                      AND GPT2 > mean(Base clean GPT2)

Qwen3-0.6B is the primary hallucination definition; GPT2 is a parallel
robustness check. WAcc is not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import analyze_failure_amplification as core


def add_dual_hallucination(
    perturbed: pd.DataFrame,
    clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    required = {"WER", "PlausibilityQwen", "PlausibilityGPT2"}
    missing_clean = required - set(clean.columns)
    missing_stress = required - set(perturbed.columns)
    if missing_clean or missing_stress:
        raise ValueError(
            "Dual-LM hallucination analysis requires WER, Qwen3 and GPT2 in both "
            f"clean and perturbed data. Missing clean={sorted(missing_clean)}, "
            f"perturbed={sorted(missing_stress)}. Run rescore_dual_lm_outputs.py first."
        )

    base = clean[clean["condition"] == "Base"].copy()
    if base.empty:
        raise ValueError("Dual-LM hallucination thresholds require Base clean rows")

    wer_t = float(pd.to_numeric(base["WER"], errors="coerce").mean())
    qwen_t = float(pd.to_numeric(base["PlausibilityQwen"], errors="coerce").mean())
    gpt2_t = float(pd.to_numeric(base["PlausibilityGPT2"], errors="coerce").mean())
    if not all(np.isfinite([wer_t, qwen_t, gpt2_t])):
        raise ValueError("Non-finite Base clean WER/Qwen/GPT2 thresholds")

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        wer = pd.to_numeric(out["WER"], errors="coerce")
        qwen = pd.to_numeric(out["PlausibilityQwen"], errors="coerce")
        gpt2 = pd.to_numeric(out["PlausibilityGPT2"], errors="coerce")

        valid_q = wer.notna() & qwen.notna()
        valid_g = wer.notna() & gpt2.notna()
        hq = pd.Series(np.nan, index=out.index, dtype=float)
        hg = pd.Series(np.nan, index=out.index, dtype=float)
        hq.loc[valid_q] = ((wer.loc[valid_q] > wer_t) & (qwen.loc[valid_q] > qwen_t)).astype(float)
        hg.loc[valid_g] = ((wer.loc[valid_g] > wer_t) & (gpt2.loc[valid_g] > gpt2_t)).astype(float)
        out["HallucinationQwen"] = hq
        out["HallucinationGPT2"] = hg
        return out

    meta = {
        "primary": "Qwen3-0.6B",
        "robustness": "GPT2",
        "shared_wer_threshold": wer_t,
        "qwen_plausibility_threshold": qwen_t,
        "gpt2_plausibility_threshold": gpt2_t,
        "qwen_criterion": (
            "WER > mean(Base clean WER) AND "
            "Qwen3 plausibility > mean(Base clean Qwen3 plausibility)"
        ),
        "gpt2_criterion": (
            "WER > mean(Base clean WER) AND "
            "GPT2 plausibility > mean(Base clean GPT2 plausibility)"
        ),
        "threshold_source": "matched Base clean rows",
    }
    print(
        "Dual-LM hallucination thresholds: "
        f"WER>{wer_t:.6f}, Qwen>{qwen_t:.6f}, GPT2>{gpt2_t:.6f}",
        flush=True,
    )
    return apply(perturbed), apply(clean), meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbed_glob", action="append", default=[])
    p.add_argument("--details_glob", action="append", default=[])
    p.add_argument("--clean", action="append", default=[], help="CONDITION=PATH_OR_GLOB")
    p.add_argument("--test_tsv", default=None)
    p.add_argument("--clips_dir", default=None)
    p.add_argument("--stress_max_samples", type=int, default=None)
    p.add_argument("--output_dir", default="results/failure_amplification_dual_lm")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zero_epsilon", type=float, default=1e-6)
    p.add_argument("--require_grounding_gap", action="store_true")
    args = p.parse_args()

    perturbed_paths = core.collect_globs(args.perturbed_glob + args.details_glob)
    if not perturbed_paths:
        raise FileNotFoundError("No perturbed details TSVs matched")
    if not args.clean:
        raise ValueError("At least one --clean CONDITION=PATH_OR_GLOB is required")

    manifest = None
    if args.test_tsv or args.clips_dir:
        if not args.test_tsv or not args.clips_dir:
            raise ValueError("--test_tsv and --clips_dir must be provided together")
        manifest = core.build_test_manifest(args.test_tsv, args.clips_dir, args.stress_max_samples)
        print(f"Reconstructed stress manifest: {len(manifest):,} rows", flush=True)

    perturbed = core.load_perturbed(perturbed_paths, manifest)
    clean = core.load_clean(args.clean)
    print(f"Loaded perturbed rows: {len(perturbed):,} from {len(perturbed_paths)} files", flush=True)
    print(f"Loaded clean rows: {len(clean):,}", flush=True)

    perturbed, clean, hallucination_meta = add_dual_hallucination(perturbed, clean)

    # Make both hallucination definitions first-class metrics for every existing
    # paired-delta / DID / signature routine. Remove the old single generic key
    # so the output cannot silently collapse back to one LM.
    core.METRIC_CANDIDATES.pop("Hallucination", None)
    core.METRIC_CANDIDATES["HallucinationQwen"] = ["HallucinationQwen"]
    core.METRIC_CANDIDATES["HallucinationGPT2"] = ["HallucinationGPT2"]

    metrics_available, metrics_missing = core.report_metric_availability(perturbed, clean)
    for required_metric in ["HallucinationQwen", "HallucinationGPT2"]:
        if required_metric not in metrics_available:
            raise RuntimeError(f"Required dual-LM metric missing after labeling: {required_metric}")

    if args.require_grounding_gap and "GroundingGap" in metrics_missing:
        raise ValueError(
            "GroundingGap is required but unavailable. Source files need grounding_gap "
            "or both hyp_ctc_nll and ref_ctc_nll."
        )

    paired = core.pair_with_clean(perturbed, clean)
    summary = core.summarize_deltas(paired, args.bootstrap, args.seed)
    did = core.difference_in_differences(
        paired, summary, args.bootstrap, args.seed, args.zero_epsilon
    )
    similarity = core.signature_similarity(summary)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "paired_clean_to_perturbed_deltas.csv", index=False)
    summary.to_csv(out / "perturbation_delta_summary.csv", index=False)
    did.to_csv(out / "difference_in_differences.csv", index=False)
    similarity.to_csv(out / "signature_similarity.csv", index=False)

    lm_comparison = pd.DataFrame()
    q = summary[summary.metric == "HallucinationQwen"][[
        "condition", "perturbation", "N", "delta_mean", "delta_ci_low", "delta_ci_high"
    ]].rename(columns={
        "delta_mean": "qwen_delta",
        "delta_ci_low": "qwen_ci_low",
        "delta_ci_high": "qwen_ci_high",
    })
    g = summary[summary.metric == "HallucinationGPT2"][[
        "condition", "perturbation", "N", "delta_mean", "delta_ci_low", "delta_ci_high"
    ]].rename(columns={
        "N": "N_gpt2",
        "delta_mean": "gpt2_delta",
        "delta_ci_low": "gpt2_ci_low",
        "delta_ci_high": "gpt2_ci_high",
    })
    if not q.empty and not g.empty:
        lm_comparison = q.merge(g, on=["condition", "perturbation"], how="outer")
        lm_comparison["qwen_minus_gpt2_delta"] = (
            lm_comparison["qwen_delta"] - lm_comparison["gpt2_delta"]
        )
        lm_comparison.to_csv(out / "hallucination_qwen_gpt2_comparison.csv", index=False)

    payload = {
        "analysis": "structured mismatch / acoustic stress with dual-LM hallucination labels",
        "hallucination_definition": hallucination_meta,
        "primary_hallucination_metric": "HallucinationQwen",
        "robustness_hallucination_metric": "HallucinationGPT2",
        "metrics_available": metrics_available,
        "metrics_missing": metrics_missing,
        "interpretation_counts": (
            {str(k): int(v) for k, v in did.interpretation.value_counts().to_dict().items()}
            if not did.empty else {}
        ),
        "n_paired_rows": int(len(paired)),
        "wacc_used": False,
    }
    (out / "hypothesis_check_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== Dual-LM failure-signature analysis ===", flush=True)
    print(f"Paired perturbed rows: {len(paired):,}", flush=True)
    if not did.empty:
        print(did.interpretation.value_counts().to_string(), flush=True)
    if not similarity.empty:
        hall_sig = similarity[
            similarity.metric.isin(["HallucinationQwen", "HallucinationGPT2"])
        ]
        if not hall_sig.empty:
            print("\nQwen/GPT2 hallucination signature similarity:", flush=True)
            print(hall_sig.sort_values(["metric", "condition"]).to_string(index=False), flush=True)
    print(f"\nOutputs written to: {out}", flush=True)


if __name__ == "__main__":
    main()
