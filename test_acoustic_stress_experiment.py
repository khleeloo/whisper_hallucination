import numpy as np
import pandas as pd

from acoustic_stress_experiment import (
    add_repetition_and_collapse_features,
    apply_dual_hallucination_labels,
    concentration_stats,
    derive_matched_clean_thresholds,
    summarize_condition,
)


def _toy_rows():
    return pd.DataFrame(
        {
            "perturbation": ["none", "none", "stress", "stress"],
            "utterance_id": ["a", "b", "a", "b"],
            "reference": ["one two", "three four", "one two", "three four"],
            "hypothesis": [
                "one two",
                "three four",
                "hello hello hello hello",
                "a completely different sentence",
            ],
            "WER": [0.0, 0.2, 2.0, 1.0],
            "qwen_plaus": [0.8, 0.9, 0.95, 0.96],
            "gpt2_plaus": [0.7, 0.9, 0.95, 0.96],
        }
    )


def test_hallucination_and_repetition_are_not_conflated():
    df = add_repetition_and_collapse_features(_toy_rows())
    thresholds = derive_matched_clean_thresholds(df)
    df = apply_dual_hallucination_labels(df, thresholds)
    stress = df[df["perturbation"].eq("stress")].reset_index(drop=True)

    assert bool(stress.loc[0, "hallucination_qwen"])
    assert bool(stress.loc[0, "repetition_3_or_4"])
    assert bool(stress.loc[0, "qwen_hall_plus_rep"])
    assert not bool(stress.loc[0, "qwen_hall_only_no_rep"])

    assert bool(stress.loc[1, "hallucination_qwen"])
    assert not bool(stress.loc[1, "repetition_3_or_4"])
    assert bool(stress.loc[1, "qwen_hall_only_no_rep"])


def test_empty_outputs_do_not_count_as_mode_collapse():
    df = pd.DataFrame(
        {
            "perturbation": ["silence"] * 5,
            "reference": ["x"] * 5,
            "hypothesis": ["", "", "stock phrase", "stock phrase", "other"],
        }
    )
    df = add_repetition_and_collapse_features(df)
    stats = concentration_stats(df)
    assert np.isclose(df["empty_output"].mean(), 0.4)
    assert stats["most_common_hypothesis"] == "stock phrase"
    assert stats["most_common_hypothesis_count"] == 2
    assert np.isclose(stats["top1_hypothesis_mass_nonempty"], 2 / 3)


def test_summary_reports_nonrepetitive_hallucination_separately():
    df = add_repetition_and_collapse_features(_toy_rows())
    thresholds = derive_matched_clean_thresholds(df)
    df = apply_dual_hallucination_labels(df, thresholds)
    stress = df[df["perturbation"].eq("stress")].copy()
    summary = summarize_condition(df, stress, n_boot=200, seed=7)

    assert summary["hallucination_qwen_rate"] == 1.0
    assert summary["qwen_hall_plus_rep_rate"] == 0.5
    assert summary["qwen_hall_only_no_rep_rate"] == 0.5
    assert summary["qwen_fraction_hallucinations_with_rep"] == 0.5
