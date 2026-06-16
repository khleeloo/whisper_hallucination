import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("finetune_whisper_lora.py")
SPEC = importlib.util.spec_from_file_location("finetune_whisper_lora", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_step_values_rounds_up_for_best_model_tracking():
    save_steps, eval_steps = MODULE.resolve_step_values(1000, 2000, True)

    assert save_steps == 2000
    assert eval_steps == 2000


def test_resolve_step_values_leaves_aligned_values_unchanged():
    save_steps, eval_steps = MODULE.resolve_step_values(2000, 2000, True)

    assert save_steps == 2000
    assert eval_steps == 2000


def test_checkpoint_step_ignores_synthetic_checkpoint_names():
    assert MODULE.checkpoint_step("checkpoint-4000") == 4000
    assert MODULE.checkpoint_step("checkpoint-best") is None
    assert MODULE.checkpoint_step("checkpoint-last") is None


def test_find_best_resume_checkpoint_requires_full_trainer_state(tmp_path):
    incomplete = tmp_path / "checkpoint-4000"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").touch()
    (incomplete / "adapter_model.safetensors").touch()

    complete = tmp_path / "checkpoint-2000"
    complete.mkdir()
    (complete / "adapter_config.json").touch()
    (complete / "adapter_model.safetensors").touch()
    (complete / "trainer_state.json").touch()
    (complete / "optimizer.pt").touch()
    (complete / "scheduler.pt").touch()

    assert MODULE.find_best_resume_checkpoint(str(tmp_path)) == str(complete)


def test_find_best_resume_checkpoint_prefers_recorded_best(tmp_path):
    best = tmp_path / "checkpoint-2000"
    latest = tmp_path / "checkpoint-4000"
    for checkpoint in [best, latest]:
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").touch()
        (checkpoint / "adapter_model.safetensors").touch()
        (checkpoint / "optimizer.pt").touch()
        (checkpoint / "scheduler.pt").touch()

    (best / "trainer_state.json").write_text(
        '{"best_model_checkpoint": "checkpoint-2000"}',
        encoding="utf-8",
    )
    (latest / "trainer_state.json").write_text(
        '{"best_model_checkpoint": "checkpoint-2000"}',
        encoding="utf-8",
    )

    assert MODULE.find_best_resume_checkpoint(str(tmp_path)) == str(best)


def test_numeric_checkpoint_cleanup_keeps_only_requested_numeric_dirs(tmp_path):
    for name in ["checkpoint-1000", "checkpoint-2000", "checkpoint-best"]:
        (tmp_path / name).mkdir()

    MODULE.cleanup_numeric_checkpoints(str(tmp_path), {"checkpoint-2000"})

    assert not (tmp_path / "checkpoint-1000").exists()
    assert (tmp_path / "checkpoint-2000").exists()
    assert (tmp_path / "checkpoint-best").exists()
