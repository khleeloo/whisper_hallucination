import numpy as np
import pandas as pd

from acoustic_abstention_before_after import build_paper_summary


def test_build_paper_summary_reports_system_and_conditional_after_rates():
    detailed = pd.DataFrame(
        {
            "perturbation": ["full_noise_amp0.75_dur0.0", "none"],
            "N": [1000, 1000],
            "coverage": [0.20, 0.99],
            "abstention_rate": [0.80, 0.01],
            "mean_WER_all": [1.8, 0.15],
            "mean_WER_accepted": [0.7, 0.14],
            "hallucination_rate_before_qwen": [0.93, 0.15],
            "emitted_hallucination_incidence_qwen": [0.08, 0.14],
            "residual_hallucination_among_accepted_qwen": [0.40, 0.141414],
            "hallucination_capture_recall_qwen": [0.914, 0.067],
            "hallucination_rate_before_gpt2": [0.94, 0.14],
            "emitted_hallucination_incidence_gpt2": [0.09, 0.13],
            "residual_hallucination_among_accepted_gpt2": [0.45, 0.131313],
            "hallucination_capture_recall_gpt2": [0.904, 0.071],
        }
    )
    out = build_paper_summary(detailed)

    # Fixed scientific order: clean before severe full-noise conditions.
    assert out.iloc[0]["condition"] == "none"
    assert out.iloc[1]["condition"] == "full_noise_amp0.75_dur0.0"

    severe = out.iloc[1]
    assert np.isclose(severe["coverage_pct"], 20.0)
    assert np.isclose(severe["Qwen_H_before_pct"], 93.0)
    # System-level after rate is over all inputs; abstentions are non-emissions.
    assert np.isclose(severe["Qwen_H_after_system_pct"], 8.0)
    # Conditional risk remains separately visible among emitted transcripts.
    assert np.isclose(severe["Qwen_H_among_emitted_pct"], 40.0)
    assert np.isclose(severe["Qwen_H_capture_pct"], 91.4)
