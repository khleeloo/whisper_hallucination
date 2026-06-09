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
