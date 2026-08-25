#!/usr/bin/env python3
"""Audit paper-facing protocol claims against the implemented experiment code.

This script is inference-free. It checks the exact dataset paths, perturbation
implementation, LM score definition, CTC normalization, intervention defaults,
DEV/TEST separation, and (when cached outputs are present) cross-model utterance
alignment. It also scans the ICASSP LaTeX file for wording that would overstate
what the code actually implements.

Run on the cluster from the repository root:

    python paper_protocol_audit.py --tex paper_icassp.tex

The JSON report is written under the scratch experiment root by default.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:  # static audit still works without pandas
    pd = None

REPO = Path(__file__).resolve().parent
ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DEFAULT_REPORT = ROOT / "paper_protocol_audit.json"

CROSS_MODEL_SOURCES = {
    "Raw Whisper": ROOT / "pretrained_whisper_stress_pipeline/rescore_explore/scored_outputs_corrected.csv",
    "Adapted Whisper": ROOT / "clean_wer_rescore/scored_outputs_cleanwer.csv",
    "SeamlessM4T-v2": ROOT / "seamless_m4t_v2_stress_pipeline_fixedwer/scored_outputs.csv",
}


def read_source(name: str) -> str:
    path = REPO / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def arg_default(source: str, arg: str) -> Optional[str]:
    # Handles argparse lines of the form add_argument("--x", ..., default=value)
    m = re.search(
        rf"add_argument\(\s*[\"']--{re.escape(arg)}[\"'][^\n]*?default\s*=\s*([^,\)]+)",
        source,
    )
    return m.group(1).strip() if m else None


def quoted_constant(source: str, name: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(name)}\s*=\s*(?:Path\()?\s*[\"']([^\"']+)[\"']", source, re.M)
    return m.group(1) if m else None


def static_checks() -> Dict[str, Any]:
    abst = read_source("acoustic_abstention_mitigation.py")
    mit = read_source("mitigation_experiment.py")
    evalw = read_source("evaluate_whisper_validation.py")
    ctc = read_source("acoustic_grounding_validation.py")
    phen = read_source("phenotype_targeted_mitigation.py")
    raw = read_source("pretrained_whisper_stress_pipeline.py")
    seam = read_source("seamless_m4t_stress_pipeline.py")
    adapted = read_source("acoustic_abstention_mitigation.py")

    checks: Dict[str, Any] = {}

    # Dataset provenance.
    cv_match = re.search(r"cv-corpus-([0-9.]+)-([0-9-]+)/en", abst)
    checks["dataset"] = {
        "corpus": "Mozilla Common Voice English",
        "version": cv_match.group(1) if cv_match else None,
        "snapshot_date": cv_match.group(2) if cv_match else None,
        "test_manifest": str(ROOT.parent / "datasets/whisper_hallucination/test.tsv"),
        "clips_path_in_code": quoted_constant(abst, "CV_ROOT"),
        "dev_test_disjoint_check_implemented": "validate_disjoint" in abst,
        "test_max_samples_raw": arg_default(raw, "test_max_samples"),
        "test_max_samples_seamless": arg_default(seam, "test_max_samples"),
        "test_max_samples_adapted": arg_default(adapted, "test_max_samples"),
    }

    # Full-noise implementation is absolute additive Gaussian noise, then clipping.
    checks["noise"] = {
        "full_noise_formula_verified": (
            "torch.randn" in abst
            and ") * amplitude" in abst
            and "out = waveform + noise" in abst
            and "torch.clamp(out, -1.0, 1.0)" in abst
        ),
        "description": "full_noise adds iid standard-normal waveform noise scaled by amplitude, then clips to [-1,1]",
        "snr_normalized": False,
        "paper_safe_wording": (
            "Full-utterance noise adds iid Gaussian waveform noise scaled by an absolute amplitude "
            "(0.50 or 0.75) before clipping to [-1,1]; the stress levels are not SNR-normalized."
        ),
    }

    # LM score: exp(-mean token NLL), then hypothesis/reference ratio clipped to [0,1].
    checks["lm_plausibility"] = {
        "sentence_score_exp_neg_mean_nll_verified": "np.exp(-nll_val)" in evalw,
        "reference_normalized_ratio_verified": "hyp / (ref + 1e-8)" in mit,
        "clipped_0_1_verified": "min(1.0, max(0.0" in mit,
        "primary_model": "Qwen/Qwen3-0.6B",
        "robustness_model": "gpt2",
        "reference_free": False,
        "paper_safe_definition": (
            "For LM m, s_m(z)=exp(-mean token NLL_m(z)) and "
            "P_m=min(1, s_m(hypothesis)/(s_m(reference)+1e-8)). "
            "This is an evaluation-only reference-normalized fluency ratio, not a reference-free detector."
        ),
    }

    checks["repetition"] = {
        "rep34_incidence_verified": (
            '(out["rep3"] > 0) | (out["rep4"] > 0)' in raw
            and '(out["rep3"] > 0) | (out["rep4"] > 0)' in seam
        ),
        "definition": "Rep34 is the fraction of outputs containing at least one repeated trigram or four-gram.",
    }

    model_match = re.search(r"DEFAULT_MODEL_NAME\s*=\s*[\"']([^\"']+)", ctc)
    checks["ctc"] = {
        "model": model_match.group(1) if model_match else None,
        "token_normalization_verified": "normalized = losses / target_lengths.float()" in abst,
        "gate_direction_verified": "values <= float(threshold)" in abst,
        "paper_safe_definition": (
            "Acoustic support is the wav2vec2-CTC summed loss for the hypothesis divided by the "
            "number of CTC target tokens; the frozen gate accepts when this normalized NLL <= tau."
        ),
    }

    penalty = arg_default(phen, "repetition_penalty")
    min_cov = arg_default(phen, "collapse_min_clean_coverage")
    cap = arg_default(phen, "collapse_max_entries")
    checks["interventions"] = {
        "repetition_penalty_default": penalty,
        "penalty_fixed_before_test_in_code": "not selected from TEST performance" in phen,
        "penalty_dev_grid_implemented_here": False,
        "collapse_lexicon_dev_only_verified": 'baseline["split"].astype(str) == "dev"' in phen,
        "collapse_min_clean_coverage": min_cov,
        "collapse_max_entries": cap,
        "paper_safe_wording": (
            f"Anti-repetition decoding uses a fixed repetition penalty of {penalty}; it is not tuned on TEST. "
            f"The dominant-output lexicon is learned from stressed DEV while preserving at least "
            f"{float(min_cov)*100:.0f}% clean-DEV coverage, with at most {cap} entries."
            if penalty and min_cov and cap else None
        ),
    }

    raw_seed = arg_default(raw, "seed")
    adapted_seed = arg_default(adapted, "seed")
    seamless_seed = arg_default(seam, "seed")
    checks["seeds"] = {
        "raw_whisper_default": raw_seed,
        "adapted_whisper_default": adapted_seed,
        "seamless_default": seamless_seed,
        "exact_noise_realization_shared_all_three": len({raw_seed, adapted_seed, seamless_seed}) == 1,
        "interpretation": (
            f"Raw Whisper defaults to seed {raw_seed}, Adapted Whisper to {adapted_seed}, and "
            f"SeamlessM4T to {seamless_seed}. Because the defaults are not identical across all three "
            "systems, describe the comparison as a shared/matched stress protocol, not as identical "
            "noisy waveforms across all three systems."
        ),
    }

    return checks


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
        if "perturbation" in df.columns:
            clean = df[df["perturbation"].astype(str) == "none"]
        else:
            clean = df
        id_col = "utterance_id" if "utterance_id" in clean.columns else None
        if id_col:
            ids = set(clean[id_col].astype(str))
            id_sets[model] = ids
            result["test_clean_ids"][model] = len(ids)
        if "perturbation" in df.columns:
            result.setdefault("rows_per_condition", {})[model] = {
                str(k): int(v) for k, v in df.groupby("perturbation").size().to_dict().items()
            }

    models = sorted(id_sets)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            key = f"{a} == {b}"
            result["pairwise_same_test_ids"][key] = id_sets[a] == id_sets[b]
    return result


def manuscript_checks(tex_path: Optional[Path], checks: Dict[str, Any]) -> Dict[str, Any]:
    if tex_path is None:
        return {"checked": False}
    if not tex_path.exists():
        return {"checked": False, "error": f"missing manuscript: {tex_path}"}
    text = tex_path.read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", text.lower())
    return {
        "checked": True,
        "path": str(tex_path),
        "mentions_common_voice_22": "common voice 22.0" in flat,
        "defines_reference_normalized_lm_score": "reference-normalized" in flat,
        "states_noise_not_snr_normalized": "not snr-normalized" in flat,
        "states_penalty_fixed_before_test": (
            "fixed before test" in flat or "set before test" in flat
        ),
        "unsafe_standalone_plausibility_wording": "standalone linguistic plausibility" in flat,
        "unsafe_identical_waveform_implication": "identical noisy waveforms" in flat,
        "uses_shared_protocol_wording": "shared acoustic-stress protocol" in flat,
        "seed_warning": checks["seeds"]["interpretation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=REPO / "paper_icassp.tex")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

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
