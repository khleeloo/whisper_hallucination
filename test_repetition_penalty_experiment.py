import importlib.util
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("repetition_penalty_experiment.py")
SPEC = importlib.util.spec_from_file_location("repetition_penalty_experiment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


DUAL_PATH = Path(__file__).with_name("evaluate_dual_metric.py")
DUAL_SPEC = importlib.util.spec_from_file_location("evaluate_dual_metric", DUAL_PATH)
DUAL_MODULE = importlib.util.module_from_spec(DUAL_SPEC)
assert DUAL_SPEC.loader is not None
sys.modules[DUAL_SPEC.name] = DUAL_MODULE
DUAL_SPEC.loader.exec_module(DUAL_MODULE)


MITIGATION_PATH = Path(__file__).with_name("mitigation_experiment.py")
MITIGATION_SPEC = importlib.util.spec_from_file_location("mitigation_experiment", MITIGATION_PATH)
MITIGATION_MODULE = importlib.util.module_from_spec(MITIGATION_SPEC)
assert MITIGATION_SPEC.loader is not None
sys.modules[MITIGATION_SPEC.name] = MITIGATION_MODULE
MITIGATION_SPEC.loader.exec_module(MITIGATION_MODULE)


def _rows(condition="Base", hypothesis="hello world", wer=0.0):
    return pd.DataFrame(
        {
            "utterance_id": ["u1", "u2"],
            "condition": [condition, condition],
            "perturbation": ["none", "none"],
            "reference": ["hello world", "good day"],
            "hypothesis": [hypothesis, "good day"],
            "WER": [wer, wer],
            "qwen_plaus": [0.8, 0.9],
            "gpt2_plaus": [0.7, 0.8],
            "rep2": [1, 0],
            "rep3": [1, 0],
            "rep4": [1, 0],
            "hallucination_like": [False, True],
            "ctc_nll": [2.0, 3.0],
            "grounding_gap": [0.2, 0.4],
        }
    )


def test_decode_functions_expose_repetition_penalty():
    assert "repetition_penalty" in inspect.signature(DUAL_MODULE.transcribe_batch).parameters
    assert "repetition_penalty" in inspect.signature(MITIGATION_MODULE.run_baseline).parameters


def test_assign_splits_is_deterministic_and_utterance_level():
    baseline = pd.concat([_rows("Base"), _rows("RR")], ignore_index=True)

    first = MODULE.assign_splits(baseline, dev_fraction=0.5, seed=7)
    second = MODULE.assign_splits(baseline, dev_fraction=0.5, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert sorted(first["utterance_id"].tolist()) == ["u1", "u2"]
    assert set(first["split"]).issubset({"dev", "test"})


def test_deterministic_decode_seed_is_pair_stable_and_condition_specific():
    first = MODULE.deterministic_decode_seed(123, "RR", "full_noise_amp0.5_dur0.0")
    second = MODULE.deterministic_decode_seed(123, "RR", "full_noise_amp0.5_dur0.0")
    other = MODULE.deterministic_decode_seed(123, "UR", "full_noise_amp0.5_dur0.0")

    assert first == second
    assert first != other
    assert MODULE.deterministic_decode_seed(None, "RR", "full_noise_amp0.5_dur0.0") is None


def test_verify_reproduction_checks_keys_wers_and_hypotheses():
    baseline = pd.concat([
        _rows("Base", wer=0.128629),
        _rows("RR", wer=0.135792),
        _rows("RU", wer=0.135806),
        _rows("UR", wer=0.435395),
        _rows("UU", wer=0.138380),
    ], ignore_index=True)
    generated = baseline.copy()

    report = MODULE.verify_reproduction(generated, baseline, tolerance=0.02)

    assert report["passes"] is True
    assert report["missing_or_extra_key_count"] == 0
    assert {row["condition"]: row["raw_hypothesis_match_rate"] for row in report["comparisons"]} == {
        "Base": 1.0,
        "RR": 1.0,
        "RU": 1.0,
        "UR": 1.0,
        "UU": 1.0,
    }


def test_verify_reproduction_fails_on_key_mismatch():
    baseline = pd.concat([
        _rows("Base", wer=0.128629),
        _rows("RR", wer=0.135792),
        _rows("RU", wer=0.135806),
        _rows("UR", wer=0.435395),
        _rows("UU", wer=0.138380),
    ], ignore_index=True)
    generated = baseline.copy()
    generated.loc[0, "utterance_id"] = "different"

    report = MODULE.verify_reproduction(generated, baseline, tolerance=0.02)

    assert report["passes"] is False
    assert report["missing_or_extra_key_count"] == 2


def test_make_before_after_requires_exact_paired_keys():
    before = _rows("Base")
    after = _rows("Base")
    joined = MODULE.make_before_after(before, after, split="dev", penalty_before=1.0, penalty_after=1.1)

    assert len(joined) == 2
    assert "WER_before" in joined.columns
    assert "WER_after" in joined.columns
    assert set(joined["repetition_penalty_after"]) == {1.1}

    bad_after = after.copy()
    bad_after.loc[0, "utterance_id"] = "missing"
    with pytest.raises(ValueError, match="key mismatch"):
        MODULE.make_before_after(before, bad_after, split="dev", penalty_before=1.0, penalty_after=1.1)


def test_select_global_penalty_uses_lowest_primary_candidate():
    grid = pd.DataFrame(
        [
            {"repetition_penalty": 1.00, "condition": "Base", "perturbation": "none", "delta_WER": 0.0, "relative_Rep4_reduction": 0.0},
            {"repetition_penalty": 1.00, "condition": "RR", "perturbation": "none", "delta_WER": 0.0, "relative_Rep4_reduction": 0.05},
            {"repetition_penalty": 1.05, "condition": "Base", "perturbation": "none", "delta_WER": 0.004, "relative_Rep4_reduction": 0.0},
            {"repetition_penalty": 1.05, "condition": "RR", "perturbation": "none", "delta_WER": 0.0, "relative_Rep4_reduction": 0.31},
            {"repetition_penalty": 1.10, "condition": "Base", "perturbation": "none", "delta_WER": 0.003, "relative_Rep4_reduction": 0.0},
            {"repetition_penalty": 1.10, "condition": "RR", "perturbation": "none", "delta_WER": 0.0, "relative_Rep4_reduction": 0.45},
        ]
    )

    selection = MODULE.select_global_penalty(grid)

    assert selection["selected"] is True
    assert selection["selection_rule"] == "primary"
    assert selection["repetition_penalty"] == pytest.approx(1.05)


def test_select_global_penalty_uses_fallback_when_primary_fails():
    grid = pd.DataFrame(
        [
            {"repetition_penalty": 1.05, "condition": "Base", "perturbation": "none", "delta_WER": 0.008, "relative_Rep4_reduction": 0.0},
            {"repetition_penalty": 1.05, "condition": "RR", "perturbation": "none", "delta_WER": 0.0, "relative_Rep4_reduction": 0.20},
            {"repetition_penalty": 1.10, "condition": "Base", "perturbation": "none", "delta_WER": 0.009, "relative_Rep4_reduction": 0.0},
            {"repetition_penalty": 1.10, "condition": "RR", "perturbation": "none", "delta_WER": 0.0, "relative_Rep4_reduction": 0.25},
        ]
    )

    selection = MODULE.select_global_penalty(grid)

    assert selection["selected"] is True
    assert selection["selection_rule"] == "fallback"
    assert selection["repetition_penalty"] == pytest.approx(1.10)


def test_summarize_before_after_reports_requested_deltas():
    before = _rows("RR")
    after = before.copy()
    after["WER"] = [0.1, 0.1]
    after["rep3"] = [0, 0]
    after["rep4"] = [0, 0]
    joined = MODULE.make_before_after(before, after, split="test", penalty_before=1.0, penalty_after=1.1)

    summary = MODULE.summarize_before_after(joined, ["condition"])

    assert summary.loc[0, "condition"] == "RR"
    assert summary.loc[0, "delta_WER"] == pytest.approx(0.1)
    assert summary.loc[0, "delta_Rep4"] == pytest.approx(-0.5)
    assert summary.loc[0, "relative_Rep4_reduction"] == pytest.approx(1.0)


def test_paired_statistics_includes_mcnemar_and_wilcoxon_columns():
    before = _rows("Base")
    after = before.copy()
    after["hallucination_like"] = [False, False]
    after["grounding_gap"] = [0.1, 0.2]
    joined = MODULE.make_before_after(before, after, split="test", penalty_before=1.0, penalty_after=1.1)

    stats = MODULE.paired_statistics(joined, ["condition"], n_resamples=100, seed=1)

    assert set(stats["metric"]) == {"WER", "hallucination_like", "grounding_gap", "rep3", "rep4"}
    hallucination = stats[stats["metric"] == "hallucination_like"].iloc[0]
    grounding = stats[stats["metric"] == "grounding_gap"].iloc[0]
    assert "mcnemar_p_value" in hallucination.index
    assert "wilcoxon_p_value" in grounding.index
