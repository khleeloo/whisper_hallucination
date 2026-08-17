import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("calculate_hallucination_like_rates.py")
SPEC = importlib.util.spec_from_file_location("calculate_hallucination_like_rates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_new_rate_uses_wer_not_wacc():
    df = pd.DataFrame(
        {
            "wer": [0.05, 0.2, 0.4],
            "wacc": [0.0, 1.0, 1.0],
            "normalized_sentence_score_Qwen3-0.6B": [0.9, 0.9, 0.1],
        }
    )

    count, rate = MODULE.compute_rate(df, 0.1, "normalized_sentence_score_Qwen3-0.6B", 0.8)

    assert count == 1
    assert rate == pytest.approx(1 / 3)

    changed_wacc = df.copy()
    changed_wacc["wacc"] = [1.0, 0.0, 0.0]
    changed_count, changed_rate = MODULE.compute_rate(
        changed_wacc,
        0.1,
        "normalized_sentence_score_Qwen3-0.6B",
        0.8,
    )

    assert changed_count == count
    assert changed_rate == pytest.approx(rate)


def test_build_rates_derives_thresholds_from_clean_base_only(tmp_path):
    base = tmp_path / "base.csv"
    rr = tmp_path / "rr.csv"
    pd.DataFrame(
        {
            "wer": [0.1, 0.3],
            "wacc": [0.9, 0.0],
            "normalized_sentence_score_gpt2": [0.4, 0.8],
            "normalized_sentence_score_Qwen3-0.6B": [0.4, 0.8],
        }
    ).to_csv(base, index=False)
    pd.DataFrame(
        {
            "wer": [0.19, 0.21],
            "wacc": [0.0, 1.0],
            "normalized_sentence_score_gpt2": [0.9, 0.9],
            "normalized_sentence_score_Qwen3-0.6B": [0.9, 0.9],
        }
    ).to_csv(rr, index=False)

    rates = MODULE.build_rates({"base_ckpt14000": [base], "rr_64pct_checkpoint-9375": [rr]}, base)
    rr_row = rates[rates["Condition"] == "RR"].iloc[0]

    assert rr_row["WER_threshold_base_mean"] == pytest.approx(0.2)
    assert rr_row["Fluency_Qwen3_0_6B_threshold_base_mean"] == pytest.approx(0.6)
    assert rr_row["hallucination_like_count"] == 1
    assert "WAcc_threshold_base_mean" not in rates.columns


def test_comparison_reports_old_wacc_label_changes_without_affecting_new_rate(tmp_path):
    base = tmp_path / "base.csv"
    rr = tmp_path / "rr.csv"
    pd.DataFrame(
        {
            "wer": [0.1, 0.3],
            "wacc": [0.9, 0.7],
            "normalized_sentence_score_Qwen3-0.6B": [0.4, 0.8],
        }
    ).to_csv(base, index=False)
    pd.DataFrame(
        {
            "wer": [0.19, 0.21],
            "wacc": [0.0, 1.0],
            "normalized_sentence_score_Qwen3-0.6B": [0.9, 0.9],
        }
    ).to_csv(rr, index=False)

    comparison = MODULE.build_wacc_vs_wer_comparison({"rr_64pct_checkpoint-9375": [rr]}, base)
    row = comparison.iloc[0]

    assert row["old_wacc_hallucination_like_count"] == 1
    assert row["new_wer_hallucination_like_count"] == 1
    assert row["changed_label_count"] == 2