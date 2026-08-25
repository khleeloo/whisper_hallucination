#!/usr/bin/env python3
"""Audit paper-facing protocol claims against the implemented experiment code.

This script is inference-free. It checks dataset provenance, perturbation
implementation, LM-score definition, CTC normalization, intervention defaults,
DEV/TEST separation, cached cross-model utterance alignment, and actual cached
run seeds when experiment metadata are available.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

REPO = Path(__file__).resolve().parent
ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_REPORT = ROOT / "paper_protocol_audit.json"

CROSS_MODEL_SOURCES = {
    "Raw Whisper": ROOT / "pretrained_whisper_stress_pipeline/rescore_explore/scored_outputs_corrected.csv",
    "Adapted Whisper": ROOT / "clean_wer_rescore/scored_outputs_cleanwer.csv",
    "SeamlessM4T-v2": ROOT / "seamless_m4t_v2_stress_pipeline_fixedwer/scored_outputs.csv",
}
RUN_SEED_FILES = {
    "Raw Whisper": ROOT / "pretrained_whisper_stress_pipeline/experiment_metadata.json",
    "Adapted Whisper": ROOT / "hallucination_mitigation_acoustic_before_after/frozen_gate_threshold.json",
    "SeamlessM4T-v2": ROOT / "seamless_m4t_v2_stress_pipeline_fixedwer/experiment_metadata.json",
}


def read_source(name: str) -> str:
    path = REPO / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def arg_default(source: str, arg: str) -> Optional[str]:
    m = re.search(
        rf"add_argument\(\s*[\"']--{re.escape(arg)}[\"'][^\n]*?default\s*=\s*([^,\)]+)",
        source,
    )
    return m.group(1).strip() if m else None


def quoted_constant(source: str, name: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(name)}\s*=\s*(?:Path\()?\s*[\"']([^\"']+)[\"']", source, re.M)
    return m.group(1) if m else None


def read_seed(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict) and "seed" in obj:
        try:
            return int(obj["seed"])
        except Exception:
            return None
    return None


def seed_checks(raw: str, adapted: str, seam: str) -> Dict[str, Any]:
    defaults = {
        "Raw Whisper": arg_default(raw, "seed"),
        "Adapted Whisper": arg_default(adapted, "seed"),
        "SeamlessM4T-v2": arg_default(seam, "seed"),
    }
    actual = {name: read_seed(path) for name, path in RUN_SEED_FILES.items()}
    known = [v for v in actual.values() if v is not None]
    exact_all = len(known) == len(actual) and len(set(known)) == 1
    if len(known) == len(actual):
        interpretation = (
            f"Cached run seeds: Raw Whisper={actual['Raw Whisper']}, "
            f"Adapted Whisper={actual['Adapted Whisper']}, "
            f"SeamlessM4T-v2={actual['SeamlessM4T-v2']}. "
            + (
                "The cached runs share the same perturbation seed."
                if exact_all
                else "Because the cached run seeds are not identical across all three systems, describe the comparison as a shared/matched stress protocol, not as identical noisy waveforms across all three systems."
            )
        )
    else:
        interpretation = (
            "One or more cached run-seed metadata files are unavailable; use shared/matched stress-protocol wording and do not claim identical noisy waveforms."
        )
    return {
        "code_defaults": defaults,
        "cached_run_seeds": actual,
        "seed_metadata_sources": {k: str(v) for k, v in RUN_SEED_FILES.items()},
        "exact_noise_realization_shared_all_three_by_seed": exact_all,
        "interpretation": interpretation,
    }


def static_checks() -> Dict[str, Any]:
    abst = read_source("acoustic_abstention_mitigation.py")
    mit = read_source("mitigation_experiment.py")
    evalw = read_source("evaluate_whisper_validation.py")
    ctc = read_source("acoustic_grounding_validation.py")
    phen = read_source("phenotype_targeted_mitigation.py")
    raw = read_source("pretrained_whisper_stress_pipeline.py")
    seam = read_source("seamless_m4t_stress_pipeline.py")
    adapted = read_source("acoustic_abstention_mitigation.py")

    cv_match = re.search(r"cv-corpus-([0-9.]+)-([0-9-]+)/en", abst)
    model_match = re.search(r"DEFAULT_MODEL_NAME\s*=\s*[\"']([^\"']+)", ctc)
    penalty = arg_default(phen, "repetition_penalty")
    min_cov = arg_default(phen, "collapse_min_clean_coverage")
    cap = arg_default(phen, "collapse_max_entries")

    return {
        "dataset": {
            "corpus": "Mozilla Common Voice English",
            "version": cv_match.group(1) if cv_match else None,
            "snapshot_date": cv_match.group(2) if cv_match else None,
            "test_manifest": str(ROOT.parent / "datasets/whisper_hallucination/test.tsv"),
            "clips_path_in_code": quoted_constant(abst, "CV_ROOT"),
            "dev_test_disjoint_check_implemented": "validate_disjoint" in abst,
            "test_max_samples_raw": arg_default(raw, "test_max_samples"),
            "test_max_samples_seamless": arg_default(seam, "test_max_samples"),
            "test_max_samples_adapted": arg_default(adapted, "test_max_samples"),
        },
        "noise": {
            "full_noise_formula_verified": (
                "torch.randn" in abst
                and ") * amplitude" in abst
                and "out = waveform + noise" in abst
                and "torch.clamp(out, -1.0, 1.0)" in abst
            ),
            "description": "full_noise adds iid standard-normal waveform noise scaled by amplitude, then clips to [-1,1]",
            "snr_normalized": False,
            "paper_safe_wording": "Full-utterance noise adds iid Gaussian waveform noise scaled by an absolute amplitude (0.50 or 0.75) before clipping to [-1,1]; the stress levels are not SNR-normalized.",
        },
        "lm_plausibility": {
            "sentence_score_exp_neg_mean_nll_verified": "np.exp(-nll_val)" in evalw,
            "reference_normalized_ratio_verified": "hyp / (ref + 1e-8)" in mit,
            "clipped_0_1_verified": "min(1.0, max(0.0" in mit,
            "primary_model": "Qwen/Qwen3-0.6B",
            "robustness_model": "gpt2",
            "reference_free": False,
            "paper_safe_definition": "For LM m, s_m(z)=exp(-mean token NLL_m(z)) and P_m=min(1, s_m(hypothesis)/(s_m(reference)+1e-8)). This is an evaluation-only reference-normalized fluency ratio, not a reference-free detector.",
        },
        "repetition": {
            "rep34_incidence_verified": (
                '(out["rep3"] > 0) | (out["rep4"] > 0)' in raw
                and '(out["rep3"] > 0) | (out["rep4"] > 0)' in seam
            ),
            "definition": "Rep34 is the fraction of outputs containing at least one repeated trigram or four-gram.",
        },
        "ctc": {
            "model": model_match.group(1) if model_match else None,
            "token_normalization_verified": "normalized = losses / target_lengths.float()" in abst,
            "gate_direction_verified": "values <= float(threshold)" in abst,
            "paper_safe_definition": "Acoustic support is the wav2vec2-CTC summed loss for the hypothesis divided by the number of CTC target tokens; the frozen gate accepts when this normalized NLL <= tau.",
        },
        "interventions": {
            "repetition_penalty_default": penalty,
            "penalty_fixed_before_test_in_code": "not selected from TEST performance" in phen,
            "penalty_dev_grid_implemented_here": False,
            "collapse_lexicon_dev_only_verified": 'baseline["split"].astype(str) == "dev"' in phen,
            "collapse_min_clean_coverage": min_cov,
            "collapse_max_entries": cap,
            "paper_safe_wording": (
                f"Anti-repetition decoding uses a fixed repetition penalty of {penalty}; it is not tuned on TEST. "
                f"The dominant-output lexicon is learned from stressed DEV while preserving at least {float(min_cov)*100:.0f}% clean-DEV coverage, with at most {cap} entries."
                if penalty and min_cov and cap else None
            ),
        },
        "seeds": seed_checks(raw, adapted, seam),
    }


def cached_output_checks() -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": {}, "test_clean_ids": {}, "pairwise_same_test_ids": {}}
    if pd is None:
        result["warning"] = "pandas unavailable; cached-output checks skipped"
        return result
    id_sets: Dict[str, set[str]] = {}
    for model, path in CROSS_MODEL_SOURCES.items():
        result["available"][model] = path.exists()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"].astype(str) == "test"]
        clean = df[df["perturbation"].astype(str) == "none"] if "perturbation" in df.columns else df
        if "utterance_id" in clean.columns:
            ids = set(clean["utterance_id"].astype(str))
            id_sets[model] = ids
            result["test_clean_ids"][model] = len(ids)
        if "perturbation" in df.columns:
            result.setdefault("rows_per_condition", {})[model] = {
                str(k): int(v) for k, v in df.groupby("perturbation").size().to_dict().items()
            }
    models = sorted(id_sets)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            result["pairwise_same_test_ids"][f"{a} == {b}"] = id_sets[a] == id_sets[b]
    return result


def manuscript_checks(tex_path: Optional[Path], checks: Dict[str, Any]) -> Dict[str, Any]:
    if tex_path is None:
        return {"checked": False}
    if not tex_path.exists():
        return {"checked": False, "error": f"missing manuscript: {tex_path}"}
    flat = re.sub(r"\s+", " ", tex_path.read_text(encoding="utf-8").lower())
    return {
        "checked": True,
        "path": str(tex_path),
        "mentions_common_voice_22": "common voice 22.0" in flat,
        "defines_reference_normalized_lm_score": "reference-normalized" in flat,
        "states_noise_not_snr_normalized": "not snr-normalized" in flat,
        "states_penalty_fixed_before_test": "fixed before test" in flat or "set before test" in flat,
        "unsafe_standalone_plausibility_wording": "standalone linguistic plausibility" in flat,
        "unsafe_identical_waveform_implication": "identical noisy waveforms" in flat,
        "uses_shared_protocol_wording": "shared acoustic-stress protocol" in flat,
        "seed_warning": checks["seeds"]["interpretation"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", type=Path, default=REPO / "paper_icassp.tex")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = p.parse_args()
    checks = static_checks()
    report = {
        "static": checks,
        "cached_outputs": cached_output_checks(),
        "manuscript": manuscript_checks(args.tex, checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== ICASSP PAPER PROTOCOL AUDIT ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
