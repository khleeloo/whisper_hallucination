import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("acoustic_grounding_validation.py")
SPEC = importlib.util.spec_from_file_location("acoustic_grounding_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_input_csv_skips_existing_file_without_lm_scores(tmp_path):
    old_csv = tmp_path / "old.csv"
    lm_csv = tmp_path / "lm.csv"
    write_csv(old_csv, ["utterance_id", "reference", "hypothesis", "WER"], [])
    write_csv(lm_csv, ["utterance_id", "audio_path", "reference", "hypothesis", "WER", "qwen3_score", "gpt2_score"], [])

    assert MODULE.select_input_csv([old_csv, lm_csv], None, None) == lm_csv


def test_select_input_csv_reports_rejected_candidates(tmp_path):
    old_csv = tmp_path / "old.csv"
    write_csv(old_csv, ["utterance_id", "reference", "hypothesis", "WER", "gpt2_score"], [])

    with pytest.raises(ValueError, match="Qwen/Qwen3"):
        MODULE.select_input_csv([old_csv], None, None)


def test_select_input_csv_requires_audio_path_for_auto_selection(tmp_path):
    text_only_csv = tmp_path / "text_only.csv"
    write_csv(text_only_csv, ["utterance_id", "reference", "hypothesis", "WER", "qwen3_score", "gpt2_score"], [])

    with pytest.raises(ValueError, match="audio_path"):
        MODULE.select_input_csv([text_only_csv], None, None)


def test_standardize_columns_coalesces_mixed_qwen_columns():
    df = pd.DataFrame(
        {
            "utt_id": ["a", "b"],
            "audio_path": ["/clips/a.mp3", "/clips/b.mp3"],
            "reference": ["hello", "world"],
            "hypothesis": ["hello", "word"],
            "wer": [0.0, 0.5],
            "normalized_sentence_score_gpt2": [0.9, 0.8],
            "normalized_sentence_score_Qwen3-0.6B": [0.7, None],
            "normalized_sentence_score_Qwen3-1.7B": [None, 0.6],
        }
    )

    out, qwen_col, gpt2_col = MODULE.standardize_columns(df, None, None)

    assert out["Qwen3_score"].tolist() == pytest.approx([0.7, 0.6])
    assert out["GPT2_score"].tolist() == pytest.approx([0.9, 0.8])
    assert "Qwen3-0.6B" in qwen_col
    assert gpt2_col == "normalized_sentence_score_gpt2"


def test_validate_manifest_alignment_accepts_low_mismatch_rate():
    df = pd.DataFrame(
        {
            "reference": ["Hello, world!", "Different text"],
            "manifest_reference": ["hello world", "another sentence"],
        }
    )

    stats = MODULE.validate_manifest_alignment(df, max_mismatch_rate=0.50)

    assert stats["checked"] == 2
    assert stats["mismatches"] == 1
    assert stats["mismatch_rate"] == pytest.approx(0.5)


def test_validate_manifest_alignment_rejects_likely_row_misalignment():
    df = pd.DataFrame(
        {
            "reference": ["one", "two", "three"],
            "manifest_reference": ["alpha", "beta", "gamma"],
        }
    )

    with pytest.raises(ValueError, match="row-aligned"):
        MODULE.validate_manifest_alignment(df, max_mismatch_rate=0.25)


def test_normalize_wav2vec2_text_uses_tokenizer_normalizer_when_available():
    class FakeTokenizer:
        def _normalize(self, text):
            return f"normalized:{text}"

    assert MODULE.normalize_wav2vec2_text(FakeTokenizer(), "Hello") == "normalized:Hello"


def test_normalize_wav2vec2_text_has_stable_fallback():
    assert MODULE.normalize_wav2vec2_text(object(), "Hello, WORLD! 123") == "HELLO WORLD"


def test_match_pairs_excludes_full_silence_rows():
    df = pd.DataFrame(
        {
            "utterance_id": ["hall_silence", "hall", "control"],
            "condition": ["ur", "ur", "ur"],
            "perturbation_condition": ["ur", "ur", "ur"],
            "model_name": ["m", "m", "m"],
            "WER": [0.60, 0.62, 0.61],
            "grounding_gap": [10.0, 4.0, 1.0],
            "hyp_ctc_nll": [12.0, 6.0, 3.0],
            "hyp_ctc_token_count": [10, 10, 11],
            "is_full_silence": [True, False, False],
            "hallucination_flag": [True, True, False],
        }
    )

    result = MODULE.match_pairs(df, "hallucination_flag", seed=1)

    assert len(result.pairs) == 1
    assert result.pairs.iloc[0]["hallucination_utterance_id"] == "hall"
