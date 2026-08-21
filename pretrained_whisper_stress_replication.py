#!/usr/bin/env python3
"""Three-condition entrypoint for the untouched Whisper-large-v3 replication.

Runs exactly the apples-to-apples protocol used for the fine-tuned model:
clean, full-noise 0.50, and full-noise 0.75.  The underlying pipeline computes
corrected WER, Qwen3/GPT-2 diagnostic and strict hallucination labels,
repetition/output concentration, wav2vec2 acoustic support, and a DEV-calibrated
reference-free gate that is frozen before TEST evaluation.
"""

import pretrained_whisper_stress_pipeline as pipeline

pipeline.DEFAULT_PERTURBATIONS = [
    "none",
    "full_noise_amp0.5_dur0.0",
    "full_noise_amp0.75_dur0.0",
]

if __name__ == "__main__":
    pipeline.main()
