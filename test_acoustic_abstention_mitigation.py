import numpy as np
import pandas as pd

from acoustic_abstention_mitigation import (
    accepted_mask,
    bootstrap_delta_incidence,
    select_gate_threshold,
    summarize_group,
)


def test_accepted_mask_rejects_nonfinite_and_above_threshold():
    got = accepted_mask([0.1, 0.2, np.inf, np.nan, 0.3], 0.2)
    assert got.tolist() == [True, True, False, False, False]


def test_select_gate_threshold_respects_clean_coverage_and_stress_objective():
    # Both 0.4 and 0.9 catch all stressed hallucinations. The tie-break
    # prefers 0.9 because it preserves more stressed and clean coverage.
    dev = pd.DataFrame(
        {
            "perturbation": ["none"] * 4 + ["full_noise_amp0.5_dur0.0"] * 4,
            "ctc_support_nll": [0.1, 0.2, 0.4, 0.9, 0.3, 0.5, 1.0, 1.2],
            "hallucination_like": [False, False, False, True, False, False, True, True],
        }
    )
    tau, table = select_gate_threshold(dev, min_clean_coverage=0.75)
    assert tau == 0.9
    assert table.iloc[0]["clean_coverage"] == 1.0
    assert table.iloc[0]["stress_hallucination_capture_recall"] == 1.0


def test_bootstrap_delta_incidence_is_nonpositive_for_abstention():
    hall = np.array([False, True, True, False])
    accept = np.array([True, False, True, True])
    point, lo, hi = bootstrap_delta_incidence(hall, accept, n_boot=500, seed=7)
    assert point == -0.25
    assert lo <= point <= hi
    assert hi <= 0.0


def test_summarize_group_reports_coverage_and_residual_risk():
    group = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "perturbation": ["full_noise_amp0.5_dur0.0"] * 4,
            "hallucination_like": [False, True, True, False],
            "ctc_support_nll": [0.1, 0.9, 0.2, 0.8],
            "WER": [0.0, 1.0, 0.8, 0.2],
        }
    )
    out = summarize_group(group, threshold=0.5, n_boot=500, seed=11)
    assert out["coverage"] == 0.5
    assert out["hallucination_rate_before"] == 0.5
    assert out["emitted_hallucination_incidence"] == 0.25
    assert out["residual_hallucination_among_accepted"] == 0.5
    assert out["hallucination_capture_recall"] == 0.5
