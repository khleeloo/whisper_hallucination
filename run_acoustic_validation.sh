#!/bin/bash
# Driver for wav2vec2 CTC acoustic-grounding validation.
#
# Usage:
#   bash run_acoustic_validation.sh sanity
#   bash run_acoustic_validation.sh full
#   bash run_acoustic_validation.sh stats
#
# Optional environment overrides:
#   INPUT_CSV=/path/to/per_utterance.csv  # required for text-only CSVs; must match TEST_TSV row order if audio_path is absent
#   OUTPUT_DIR=/path/to/acoustic_validation
#   MODEL_NAME=facebook/wav2vec2-large-960h-lv60-self
#   DEVICE=auto
#   QWEN_COL=Qwen3_score
#   GPT2_COL=GPT2_score
#   MAX_MANIFEST_MISMATCH_RATE=0.25

set -euo pipefail

MODE="${1:-sanity}"

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/vemotionsys/rmfrieske/whisper_hallucination}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/rmfrieske/whisper_hallucination}"
TEST_TSV="${TEST_TSV:-/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv}"
CLIPS_DIR="${CLIPS_DIR:-/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/acoustic_validation}"
MODEL_NAME="${MODEL_NAME:-facebook/wav2vec2-large-960h-lv60-self}"
DEVICE="${DEVICE:-auto}"
MAX_MANIFEST_MISMATCH_RATE="${MAX_MANIFEST_MISMATCH_RATE:-0.25}"

cd "${WORKSPACE_ROOT}"
mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --test_tsv "${TEST_TSV}"
    --clips_dir "${CLIPS_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --model_name "${MODEL_NAME}"
    --device "${DEVICE}"
    --max_manifest_mismatch_rate "${MAX_MANIFEST_MISMATCH_RATE}"
)

if [[ -n "${INPUT_CSV:-}" ]]; then
    COMMON_ARGS+=(--input_csv "${INPUT_CSV}")
fi
if [[ -n "${QWEN_COL:-}" ]]; then
    COMMON_ARGS+=(--qwen_col "${QWEN_COL}")
fi
if [[ -n "${GPT2_COL:-}" ]]; then
    COMMON_ARGS+=(--gpt2_col "${GPT2_COL}")
fi

case "${MODE}" in
    sanity)
        python acoustic_grounding_validation.py "${COMMON_ARGS[@]}" --sanity_only --max_score_rows 20
        ;;
    full)
        python acoustic_grounding_validation.py "${COMMON_ARGS[@]}"
        ;;
    stats)
        python acoustic_grounding_validation.py "${COMMON_ARGS[@]}" --skip_scoring
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Expected one of: sanity, full, stats" >&2
        exit 2
        ;;
esac
