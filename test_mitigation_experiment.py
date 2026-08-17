import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("mitigation_experiment.py")
SPEC = importlib.util.spec_from_file_location("mitigation_experiment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
CURRENT_RESULT_FILE_GROUPS = {
    "Base": [SCRATCH_ROOT / "eval_validation" / "per_utterance_base_ckpt14000.csv"],
    "RR": [SCRATCH_ROOT / "eval_64pct" / "per_utterance_rr_64pct_checkpoint-9375.csv"],
    "RU": [SCRATCH_ROOT / "eval_64pct" / "per_utterance_ru_64pct_checkpoint-9375.csv"],
    "UR": sorted((SCRATCH_ROOT / "eval_64pct").glob("per_utterance_ur_64pct_checkpoint-10000_shard*.csv")),
    "UU": sorted((SCRATCH_ROOT / "eval_64pct").glob("per_utterance_uu_64pct_final_shard*.csv")),
}


def _load_existing_csvs(paths):
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)


def test_evaluate_hypotheses_builds_required_schema_and_flags(monkeypatch, tmp_path):
    def fake_wer(hypotheses, references):
        wers = [0.0, 0.5, 0.75, 0.25]
        return [
            {
                "wer": wer,
                "wacc": 1.0 - wer,
                "s_count": 0,
                "d_count": 0,
                "i_count": 0,
                "num_ref_words": 2,
                "num_hyp_words": 2,
            }
            for wer in wers
        ]

    def fake_lm(hypotheses, references, model_name, device, batch_size):
        if "Qwen" in model_name:
            return [0.2, 0.9, 0.95, 0.1]
        return [0.3, 0.8, 0.7, 0.2]

    def fake_ctc(df, model_name, device, cache_path):
        out = df.copy()
        out["hyp_ctc_nll"] = [1.0, 2.0, 3.0, 4.0]
        out["grounding_gap"] = [0.1, 0.2, 0.3, 0.4]
        assert "perturbation" in out.columns
        return out

    monkeypatch.setattr(MODULE, "compute_wer_metrics", fake_wer)
    thresholds = {"wer_threshold": 0.4, "qwen_plausibility_threshold": 0.6}

    result = MODULE.evaluate_hypotheses(
        utterance_ids=["u0", "u1", "u2", "u3"],
        conditions=["Base", "Base", "Base", "Base"],
        perturbations=["none", "none", "none", "none"],
        references=["hello world"] * 4,
        hypotheses=["hello world", "bad transcript", "echo echo echo echo", ""],
        audio_paths=[str(tmp_path / f"{idx}.wav") for idx in range(4)],
        device="cpu",
        lm_score_fn=fake_lm,
        ctc_score_fn=fake_ctc,
        hallucination_thresholds=thresholds,
    )

    assert list(result.columns) == MODULE.OUTPUT_COLUMNS
    assert result["WER"].tolist() == pytest.approx([0.0, 0.5, 0.75, 0.25])
    assert result["qwen_plaus"].tolist() == pytest.approx([0.2, 0.9, 0.95, 0.1])
    assert result["gpt2_plaus"].tolist() == pytest.approx([0.3, 0.8, 0.7, 0.2])
    assert result["rep2"].tolist()[2] > 0
    assert result["rep3"].tolist()[2] > 0
    assert result["rep4"].tolist()[2] == 0
    assert result["hallucination_like"].tolist() == [False, True, True, False]
    assert result["ctc_nll"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result["grounding_gap"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_frozen_thresholds_do_not_change_by_condition_or_perturbation(monkeypatch, tmp_path):
    def fake_wer(hypotheses, references):
        wers = [0.5, 0.5, 0.3, 0.3]
        return [
            {
                "wer": wer,
                "wacc": 1.0 - wer,
                "s_count": 0,
                "d_count": 0,
                "i_count": 0,
                "num_ref_words": 2,
                "num_hyp_words": 2,
            }
            for wer in wers
        ]

    def fake_lm(hypotheses, references, model_name, device, batch_size):
        return [0.8, 0.8, 0.8, 0.8]

    monkeypatch.setattr(MODULE, "compute_wer_metrics", fake_wer)
    thresholds = {"wer_threshold": 0.4, "qwen_plausibility_threshold": 0.6}

    result = MODULE.evaluate_hypotheses(
        utterance_ids=["base_clean", "rr_noise", "ur_clean", "uu_noise"],
        conditions=["Base", "RR", "UR", "UU"],
        perturbations=["none", "full_noise_amp0.5_dur0.0", "none", "full_noise_amp0.5_dur0.0"],
        references=["hello world"] * 4,
        hypotheses=["x"] * 4,
        audio_paths=[str(tmp_path / f"{idx}.wav") for idx in range(4)],
        device="cpu",
        lm_score_fn=fake_lm,
        score_ctc=False,
        hallucination_thresholds=thresholds,
    )

    assert result["hallucination_like"].tolist() == [True, True, False, False]


def test_evaluate_hypotheses_validates_parallel_inputs(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        MODULE.evaluate_hypotheses(
            utterance_ids=["u0"],
            conditions=["Base", "RR"],
            perturbations=["none"],
            references=["hello"],
            hypotheses=["hello"],
            audio_paths=[str(tmp_path / "u0.wav")],
            score_lms=False,
            score_ctc=False,
        )


def test_parse_and_select_perturbations():
    parsed = MODULE.parse_perturbation("speech_band_noise_amp0.75_dur0.0")

    assert parsed.label == "speech_band_noise_amp0.75_dur0.0"
    assert parsed.perturb_type == "speech_band_noise"
    assert parsed.amplitude == pytest.approx(0.75)
    assert parsed.duration == pytest.approx(0.0)
    assert [item.label for item in MODULE.selected_perturbations(["none", parsed.label])] == [
        "none",
        parsed.label,
    ]


def test_aggregate_baseline_uses_required_output_columns():
    df = pd.DataFrame(
        {
            "utterance_id": ["a", "b"],
            "condition": ["Base", "Base"],
            "perturbation": ["none", "none"],
            "reference": ["x", "y"],
            "hypothesis": ["x", "z"],
            "WER": [0.0, 0.5],
            "qwen_plaus": [0.5, 0.9],
            "gpt2_plaus": [0.4, 0.8],
            "rep2": [0, 1],
            "rep3": [0, 0],
            "rep4": [0, 0],
            "hallucination_like": [False, True],
            "ctc_nll": [1.0, 2.0],
            "grounding_gap": [0.1, 0.3],
        }
    )

    aggregate = MODULE.aggregate_baseline(df)

    assert aggregate.loc[0, "n_samples"] == 2
    assert aggregate.loc[0, "mean_WER"] == pytest.approx(0.25)
    assert aggregate.loc[0, "mean_WAcc"] == pytest.approx(0.75)
    assert aggregate.loc[0, "hallucination_like_rate"] == pytest.approx(0.5)


def test_compute_frozen_thresholds_from_clean_base_writes_json(tmp_path):
    source = tmp_path / "per_utterance_base_ckpt14000.csv"
    output = tmp_path / "frozen_hallucination_thresholds.json"
    pd.DataFrame(
        {
            "wer": [0.1, 0.3, 0.5],
            "normalized_sentence_score_Qwen3-0.6B": [0.2, 0.4, 1.0],
        }
    ).to_csv(source, index=False)

    thresholds = MODULE.compute_frozen_hallucination_thresholds(source, output)

    assert thresholds["source_condition"] == "Base"
    assert thresholds["criterion_name"] == MODULE.DEFAULT_CRITERION_NAME
    assert thresholds["wer_threshold"] == pytest.approx((0.1 + 0.3 + 0.5) / 3.0)
    assert thresholds["qwen_plausibility_threshold"] == pytest.approx((0.2 + 0.4 + 1.0) / 3.0)
    assert thresholds["N"] == 3
    assert "base_mean_wacc" not in thresholds
    assert "wacc_source_column" not in thresholds
    assert output.exists()


def test_thresholds_are_clean_base_only_on_current_result_files():
    missing_conditions = [
        condition for condition, paths in CURRENT_RESULT_FILE_GROUPS.items() if _load_existing_csvs(paths).empty
    ]
    if missing_conditions:
        pytest.skip(f"Missing current paper per-utterance files for: {missing_conditions}")

    base = _load_existing_csvs(CURRENT_RESULT_FILE_GROUPS["Base"])
    qwen_col = MODULE._find_qwen_plausibility_column(base)
    thresholds = MODULE.compute_frozen_hallucination_thresholds(
        CURRENT_RESULT_FILE_GROUPS["Base"][0],
        Path("/tmp") / "whisper_hallucination_test_thresholds.json",
    )

    assert thresholds["criterion_name"] == "mean_WER_mean_Qwen3_clean_base"
    assert thresholds["wer_threshold"] == pytest.approx(float(base["wer"].astype(float).mean()))
    assert thresholds["qwen_plausibility_threshold"] == pytest.approx(float(base[qwen_col].astype(float).mean()))
    assert "wacc" not in " ".join(thresholds.keys()).lower()

    row_count = 0
    for paths in CURRENT_RESULT_FILE_GROUPS.values():
        df = _load_existing_csvs(paths)
        result = MODULE.apply_frozen_hallucination_thresholds(
            pd.DataFrame({"WER": df["wer"], "qwen_plaus": df[qwen_col]}),
            thresholds,
        )
        row_count += len(df)
        assert result.dtype == bool

    assert row_count > 0