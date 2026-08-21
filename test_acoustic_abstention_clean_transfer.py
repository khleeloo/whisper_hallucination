import numpy as np
import pandas as pd

from acoustic_abstention_clean_transfer import (
    binary_ranking_metrics,
    summarize_condition,
    threshold_for_min_coverage,
)


def test_threshold_for_min_coverage_uses_smallest_feasible_tau():
    tau, realized = threshold_for_min_coverage([0.1, 0.2, 0.3, 0.4, 0.5], 0.8)
    assert np.isclose(tau, 0.4)
    assert np.isclose(realized, 0.8)


def test_threshold_for_min_coverage_handles_ties_conservatively():
    tau, realized = threshold_for_min_coverage([0.1, 0.2, 0.2, 0.2, 0.9], 0.6)
    assert np.isclose(tau, 0.2)
    assert np.isclose(realized, 0.8)


def test_binary_ranking_metrics_are_perfect_for_perfect_separation():
    out = binary_ranking_metrics([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert np.isclose(out["roc_auc"], 1.0)
    assert np.isclose(out["average_precision"], 1.0)
    assert out["positives"] == 2


def test_summarize_condition_reports_transfer_metrics():
    group = pd.DataFrame(
        {
            "hallucination_like_qwen": [False, True, True, False],
            "ctc_support_nll": [0.1, 0.9, 0.2, 0.8],
            "WER": [0.0, 1.0, 0.8, 0.2],
        }
    )
    out = summarize_condition(group, tau=0.5, lm="qwen")
    assert np.isclose(out["coverage"], 0.5)
    assert np.isclose(out["hallucination_rate_before"], 0.5)
    assert np.isclose(out["system_emitted_hallucination_incidence"], 0.25)
    assert np.isclose(out["hallucination_rate_among_emitted"], 0.5)
    assert np.isclose(out["hallucination_capture_recall"], 0.5)
    assert np.isclose(out["rejection_precision_for_hallucination"], 0.5)
