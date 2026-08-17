import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("prepare_tfidf_acoustic_input.py")
SPEC = importlib.util.spec_from_file_location("prepare_tfidf_acoustic_input", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_acoustic_input_attaches_audio_by_detail_row():
    candidates = pd.DataFrame(
        {
            "detail_row": [1],
            "reference": ["She'll be all right."],
            "hypothesis": ["she will be all right"],
            "wacc": [0.75],
            "norm_plausibility": [0.6],
            "eval_config": ["base"],
            "perturbation": ["none"],
            "2gram_reps": [1],
            "3gram_reps": [0],
            "4gram_reps": [0],
        }
    )
    manifest = pd.DataFrame(
        {
            "detail_row": [0, 1],
            "audio_path": ["clips/a.mp3", "clips/b.mp3"],
            "manifest_path": ["a.mp3", "b.mp3"],
            "manifest_sentence": ["other", "she ll be all right"],
            "manifest_sentence_id": ["a", "b"],
            "manifest_client_id": ["client-a", "client-b"],
        }
    )

    out = MODULE.build_acoustic_input(candidates, manifest, max_mismatch_rate=0.0)

    assert out.loc[0, "audio_path"] == "clips/b.mp3"
    assert out.loc[0, "utterance_id"] == "base_none_000001"
    assert out.loc[0, "WER"] == pytest.approx(0.25)
    assert out.loc[0, "Qwen3_score"] == pytest.approx(0.6)
    assert out.loc[0, "GPT2_score"] == pytest.approx(0.6)
    assert out.loc[0, "bigram_rep_count"] == 1


def test_build_acoustic_input_rejects_wrong_manifest():
    candidates = pd.DataFrame(
        {
            "detail_row": [0],
            "reference": ["expected sentence"],
            "hypothesis": ["hyp"],
            "wacc": [0.5],
            "norm_plausibility": [0.7],
            "eval_config": ["base"],
            "perturbation": ["none"],
        }
    )
    manifest = pd.DataFrame(
        {
            "detail_row": [0],
            "audio_path": ["clips/a.mp3"],
            "manifest_path": ["a.mp3"],
            "manifest_sentence": ["different sentence"],
            "manifest_sentence_id": ["a"],
            "manifest_client_id": ["client-a"],
        }
    )

    with pytest.raises(ValueError, match="alignment failed"):
        MODULE.build_acoustic_input(candidates, manifest, max_mismatch_rate=0.0)
