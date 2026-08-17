import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("prepare_eval_validation_acoustic_input.py")
SPEC = importlib.util.spec_from_file_location("prepare_eval_validation_acoustic_input", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_per_utterance(path, model_name):
    pd.DataFrame(
        {
            "utt_id": [f"{model_name}_utt"],
            "audio_path": [f"/clips/{model_name}.mp3"],
            "reference": ["hello"],
            "hypothesis": ["hello"],
            "wer": [0.0],
            "model_name": [model_name],
        }
    ).to_csv(path, index=False)


def test_merge_eval_validation_concatenates_per_model_files(tmp_path):
    write_per_utterance(tmp_path / "per_utterance_base.csv", "base")
    write_per_utterance(tmp_path / "per_utterance_rr.csv", "rr")
    write_per_utterance(tmp_path / "per_utterance_metrics_whisper.csv", "old_merged")

    merged = MODULE.merge_eval_validation(tmp_path)

    assert list(merged["model_name"]) == ["base", "rr"]
    assert "source_eval_csv" in merged.columns


def test_merge_eval_validation_requires_audio_path(tmp_path):
    pd.DataFrame(
        {
            "utt_id": ["utt"],
            "reference": ["hello"],
            "hypothesis": ["hello"],
            "wer": [0.0],
        }
    ).to_csv(tmp_path / "per_utterance_base.csv", index=False)

    with pytest.raises(ValueError, match="audio_path"):
        MODULE.merge_eval_validation(tmp_path)
