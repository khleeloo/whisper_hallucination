import csv
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).with_name("evaluate_whisper_validation.py")
SPEC = importlib.util.spec_from_file_location("evaluate_whisper_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_shard_partitions_samples_without_overlap():
    samples = list(range(25))
    shards = [MODULE.select_shard(samples, shard_id, 3) for shard_id in range(3)]

    flattened = [item for shard in shards for item in shard]

    assert sorted(flattened) == samples
    assert len(flattened) == len(set(flattened))
    assert shards[0] == samples[0::3]
    assert shards[1] == samples[1::3]
    assert shards[2] == samples[2::3]


def test_select_shard_validates_bounds():
    with pytest.raises(ValueError, match="num_shards"):
        MODULE.select_shard([], 0, 0)

    with pytest.raises(ValueError, match="shard_id"):
        MODULE.select_shard([], 2, 2)


def test_resolve_output_suffix_preserves_unsharded_default():
    assert MODULE.resolve_output_suffix(None, 0, 1) == ""


def test_resolve_output_suffix_derives_shard_suffix():
    assert MODULE.resolve_output_suffix(None, 1, 2) == "_shard01-of-02"


def test_resolve_output_suffix_accepts_safe_custom_suffix():
    assert MODULE.resolve_output_suffix("_gpu0", 0, 2) == "_gpu0"


def test_resolve_output_suffix_rejects_path_components():
    with pytest.raises(ValueError, match="path separators"):
        MODULE.resolve_output_suffix("nested/shard0", 0, 2)


def test_resolve_output_paths_keeps_normal_and_gated_separate(tmp_path):
    normal_csv, normal_json = MODULE.resolve_output_paths(str(tmp_path), "uu", "_shard00", "normal")
    gated_csv, gated_json = MODULE.resolve_output_paths(str(tmp_path), "uu", "_shard00", "gated")

    assert normal_csv.endswith("per_utterance_uu_shard00.csv")
    assert normal_json.endswith("summary_uu_shard00.json")
    assert gated_csv.endswith("per_utterance_uu_shard00_gated.csv")
    assert gated_json.endswith("summary_uu_shard00_gated.json")
    assert normal_csv != gated_csv
    assert normal_json != gated_json


def test_load_test_data_reads_existing_clip_rows(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.mp3").touch()
    (clips_dir / "b.wav").touch()

    tsv_path = tmp_path / "test.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"path": "a.mp3", "sentence": "hello"})
        writer.writerow({"path": "missing.mp3", "sentence": "skip"})
        writer.writerow({"path": "b.wav", "sentence": "world"})

    samples = MODULE.load_test_data(str(tsv_path), str(clips_dir))

    assert [sample["utt_id"] for sample in samples] == ["a", "b"]
    assert [sample["reference"] for sample in samples] == ["hello", "world"]


def test_compute_avg_logprobs_from_generate_gathers_generated_tokens():
    gen_out = SimpleNamespace(
        sequences=torch.tensor([
            [99, 1, 2],
            [99, 2, 1],
        ]),
        scores=(
            torch.tensor([
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]),
            torch.tensor([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]),
        ),
    )

    actual = MODULE.compute_avg_logprobs_from_generate(gen_out)
    expected = []
    for row_idx, tokens in enumerate([[1, 2], [2, 1]]):
        values = []
        for step_idx, token_id in enumerate(tokens):
            values.append(torch.log_softmax(gen_out.scores[step_idx][row_idx], dim=-1)[token_id].item())
        expected.append(sum(values) / len(values))

    assert actual == pytest.approx(expected)


def test_compute_compression_ratio_uses_gzip_byte_formula():
    text = "the city the city the city the city"
    expected = len(text.encode("utf-8")) / len(gzip.compress(text.encode("utf-8")))

    assert MODULE.compute_compression_ratio(text) == pytest.approx(expected)
    assert MODULE.compute_compression_ratio("") == 0.0
    assert MODULE.compute_compression_ratio("repeat repeat repeat repeat repeat repeat") > MODULE.compute_compression_ratio(
        "alpha bravo charlie delta echo foxtrot"
    )


def test_calibrate_gate_thresholds_uses_percentiles_and_ignores_nonfinite():
    thresholds = MODULE.calibrate_gate_thresholds(
        [-10.0, -8.0, float("nan"), -6.0, -4.0, -2.0],
        [1.0, 2.0, float("inf"), 3.0, 4.0, 5.0],
    )

    assert thresholds["T_logprob"] == pytest.approx(
        MODULE.np.percentile([-10.0, -8.0, -6.0, -4.0, -2.0], 5)
    )
    assert thresholds["T_compression"] == pytest.approx(MODULE.np.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95))


def test_gate_threshold_json_and_gzip_round_trip(tmp_path):
    thresholds = {"T_logprob": -2.5, "T_compression": 1.75}
    json_path = tmp_path / "thresholds.json"
    gzip_path = tmp_path / "thresholds.json.gz"

    MODULE.save_gate_thresholds(thresholds, str(json_path))
    MODULE.save_gate_thresholds(thresholds, str(gzip_path))

    assert json.loads(json_path.read_text(encoding="utf-8")) == thresholds
    assert MODULE.load_gate_thresholds(str(json_path)) == thresholds
    assert MODULE.load_gate_thresholds(str(gzip_path)) == thresholds


def test_apply_gate_signals_uses_or_for_combined_gate():
    thresholds = {"T_logprob": -5.0, "T_compression": 2.0}

    assert MODULE.apply_gate_signals(-6.0, 1.0, 0, thresholds)["combined_gate"] is True
    assert MODULE.apply_gate_signals(-4.0, 2.5, 0, thresholds)["combined_gate"] is True
    assert MODULE.apply_gate_signals(-4.0, 1.0, 1, thresholds)["combined_gate"] is True
    assert MODULE.apply_gate_signals(-4.0, 1.0, 0, thresholds)["combined_gate"] is False


def test_summarize_gate_ablation_reports_before_after_metrics():
    flags = [
        {"avg_logprob_only": True, "compression_ratio_only": False, "fourgram_repetition_only": False, "combined_gate": True},
        {"avg_logprob_only": False, "compression_ratio_only": False, "fourgram_repetition_only": False, "combined_gate": False},
        {"avg_logprob_only": False, "compression_ratio_only": True, "fourgram_repetition_only": False, "combined_gate": True},
        {"avg_logprob_only": False, "compression_ratio_only": False, "fourgram_repetition_only": False, "combined_gate": False},
    ]
    summary = MODULE.summarize_gate_ablation(
        flags,
        hallucination_like=[True, False, False, True],
        wer=[1.0, 0.1, 0.4, 0.8],
        bleu=[0.0, 0.9, 0.4, 0.2],
    )

    combined = summary["combined_gate"]
    assert combined["n_samples"] == 4
    assert combined["hallucination_like_rate_before_gate"] == pytest.approx(0.5)
    assert combined["gate_flag_rate"] == pytest.approx(0.5)
    assert combined["accepted_fraction"] == pytest.approx(0.5)
    assert combined["hallucination_recall"] == pytest.approx(0.5)
    assert combined["gate_precision"] == pytest.approx(0.5)
    assert combined["false_positive_rate"] == pytest.approx(0.5)
    assert combined["hallucination_like_rate_after_gate_among_accepted"] == pytest.approx(0.5)
    assert combined["WER_before_gate"] == pytest.approx(0.575)
    assert combined["WER_after_gate_among_accepted"] == pytest.approx(0.45)
    assert combined["BLEU_before_gate"] == pytest.approx(0.375)
    assert combined["BLEU_after_gate_among_accepted"] == pytest.approx(0.55)


def test_transcribe_batch_requests_scores_only_in_gated_mode(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "load_audio_features",
        lambda path, feature_extractor: torch.ones(80, 3000),
    )

    class FakeTokenizer:
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["hello", "world"]

    class FakeProcessor:
        feature_extractor = object()
        tokenizer = FakeTokenizer()

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, input_features, **kwargs):
            self.calls.append(kwargs)
            sequences = torch.tensor([[99, 1, 2], [99, 2, 1]])
            if kwargs.get("return_dict_in_generate"):
                return SimpleNamespace(
                    sequences=sequences,
                    scores=(
                        torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
                        torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                    ),
                )
            return sequences

    normal_model = FakeModel()
    normal_hypotheses = MODULE.transcribe_batch(
        normal_model, FakeProcessor(), ["a.wav", "b.wav"], device="cpu", batch_size=2
    )
    assert normal_hypotheses == ["hello", "world"]
    assert "output_scores" not in normal_model.calls[0]
    assert "return_dict_in_generate" not in normal_model.calls[0]

    gated_model = FakeModel()
    gated_hypotheses, avg_logprobs = MODULE.transcribe_batch(
        gated_model,
        FakeProcessor(),
        ["a.wav", "b.wav"],
        device="cpu",
        batch_size=2,
        return_decoder_signals=True,
    )
    assert gated_hypotheses == ["hello", "world"]
    assert gated_model.calls[0]["output_scores"] is True
    assert gated_model.calls[0]["return_dict_in_generate"] is True
    assert avg_logprobs == pytest.approx(
        MODULE.compute_avg_logprobs_from_generate(
            SimpleNamespace(
                sequences=torch.tensor([[99, 1, 2], [99, 2, 1]]),
                scores=(
                    torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
                    torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                ),
            )
        )
    )