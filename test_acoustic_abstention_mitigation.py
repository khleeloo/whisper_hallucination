import numpy as np
import pandas as pd

from acoustic_abstention_mitigation import (
    accepted_mask,
    apply_hallucination_labels,
    bootstrap_delta_incidence,
    derive_hallucination_thresholds,
    select_gate_threshold,
    summarize_group,
)


def test_accepted_mask_rejects_nonfinite_and_above_threshold():
    got = accepted_mask([0.1, 0.2, np.inf, np.nan, 0.3], 0.2)
    assert got.tolist() == [True, True, False, False, False]


def test_dual_lm_thresholds_come_from_clean_dev_only():
    scored = pd.DataFrame(
        {
            "split": ["dev", "dev", "dev", "test"],
            "perturbation": ["none", "none", "full_noise_amp0.5_dur0.0", "none"],
            "WER": [0.1, 0.3, 9.0, 8.0],
            "qwen_plaus": [0.8, 1.0, 0.1, 0.2],
            "gpt2_plaus": [0.7, 0.9, 0.2, 0.1],
        }
    )
    t = derive_hallucination_thresholds(scored)
    assert np.isclose(t["wer_threshold"], 0.2)
    assert np.isclose(t["qwen_plausibility_threshold"], 0.9)
    assert np.isclose(t["gpt2_plausibility_threshold"], 0.8)
    assert t["N_clean_dev"] == 2


def test_hallucination_labels_keep_qwen_primary_and_gpt2_parallel():
    scored = pd.DataFrame(
        {
            "WER": [0.4, 0.4, 0.1],
            "qwen_plaus": [0.95, 0.4, 0.95],
            "gpt2_plaus": [0.4, 0.95, 0.95],
        }
    )
    t = {
        "wer_threshold": 0.2,
        "qwen_plausibility_threshold": 0.8,
        "gpt2_plausibility_threshold": 0.8,
    }
    out = apply_hallucination_labels(scored, t)
    assert out["hallucination_like_qwen"].tolist() == [True, False, False]
    assert out["hallucination_like_gpt2"].tolist() == [False, True, False]
    assert out["hallucination_like"].tolist() == [True, False, False]


def test_select_gate_threshold_respects_clean_coverage_and_qwen_primary_objective():
    # Both 0.4 and 0.9 catch all Qwen-primary stressed hallucinations. The tie-break
    # prefers 0.9 because it preserves more stressed and clean coverage.
    dev = pd.DataFrame(
        {
            "perturbation": ["none"] * 4 + ["full_noise_amp0.5_dur0.0"] * 4,
            "ctc_support_nll": [0.1, 0.2, 0.4, 0.9, 0.3, 0.5, 1.0, 1.2],
            "hallucination_like_qwen": [False, False, False, True, False, False, True, True],
            "hallucination_like_gpt2": [False, False, True, False, False, True, False, True],
        }
    )
    tau, table = select_gate_threshold(dev, min_clean_coverage=0.75)
    assert tau == 0.9
    assert table.iloc[0]["clean_coverage"] == 1.0
    assert table.iloc[0]["stress_hallucination_capture_recall_qwen"] == 1.0
    assert "stress_hallucination_capture_recall_gpt2" in table.columns


def test_bootstrap_delta_incidence_is_nonpositive_for_abstention():
    hall = np.array([False, True, True, False])
    accept = np.array([True, False, True, True])
    point, lo, hi = bootstrap_delta_incidence(hall, accept, n_boot=500, seed=7)
    assert point == -0.25
    assert lo <= point <= hi
    assert hi <= 0.0


def test_summarize_group_reports_both_lm_labels():
    group = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "perturbation": ["full_noise_amp0.5_dur0.0"] * 4,
            "hallucination_like": [False, True, True, False],
            "hallucination_like_qwen": [False, True, True, False],
            "hallucination_like_gpt2": [True, False, True, False],
            "ctc_support_nll": [0.1, 0.9, 0.2, 0.8],
            "WER": [0.0, 1.0, 0.8, 0.2],
        }
    )
    out = summarize_group(group, threshold=0.5, n_boot=500, seed=11)
    assert out["coverage"] == 0.5
    assert out["hallucination_rate_before_qwen"] == 0.5
    assert out["emitted_hallucination_incidence_qwen"] == 0.25
    assert out["hallucination_capture_recall_qwen"] == 0.5
    assert out["hallucination_rate_before_gpt2"] == 0.5
    assert out["emitted_hallucination_incidence_gpt2"] == 0.5
    assert out["hallucination_rate_before"] == out["hallucination_rate_before_qwen"]
