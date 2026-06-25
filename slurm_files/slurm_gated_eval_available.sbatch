#!/bin/bash
#SBATCH --job-name=wh_gated_eval
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --array=0-6
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_wh_gated_eval.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_wh_gated_eval.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
TEST_TSV_64="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
TEST_TSV_100="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct/test.tsv"
OUTPUT_DIR="${BASE_DIR}/eval_gated_all"

mkdir -p "${OUTPUT_DIR}" /home/rmfrieske/whisper_hallucination/slurm_logs

CONFIG_NAMES=(
  "base_checkpoint-10000"
  "uu_64pct_final"
  "ur_64pct_checkpoint-10000"
  "rr_64pct_checkpoint-9375"
  "ru_64pct_checkpoint-9375"
  "uu_100pct_final"
  "ur_100pct_checkpoint-4000"
)

MODEL_DIRS=(
  "${BASE_DIR}/base/checkpoint-10000"
  "${BASE_DIR}/uu_64pct/final"
  "${BASE_DIR}/ur_64pct/checkpoint-10000"
  "${BASE_DIR}/rr_64pct/checkpoint-9375"
  "${BASE_DIR}/ru_64pct/checkpoint-9375"
  "${BASE_DIR}/uu_100pct/final"
  "${BASE_DIR}/ur_100pct/checkpoint-4000"
)

NOISE_CONDITIONS=("base" "UU" "UR" "RR" "RU" "UU" "UR")
NOISE_RATIOS=("0.0" "0.64" "0.64" "0.64" "0.64" "1.0" "1.0")
TEST_TSVS=(
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_100}"
  "${TEST_TSV_100}"
)

idx="${SLURM_ARRAY_TASK_ID}"
CONFIG_NAME="${CONFIG_NAMES[$idx]}"
MODEL_DIR="${MODEL_DIRS[$idx]}"
NOISE_CONDITION="${NOISE_CONDITIONS[$idx]}"
NOISE_RATIO="${NOISE_RATIOS[$idx]}"
TEST_TSV="${TEST_TSVS[$idx]}"

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
  echo "ERROR: missing adapter_config.json in ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
  echo "ERROR: missing adapter weights in ${MODEL_DIR}" >&2
  exit 1
fi

echo "=== Gated eval: ${CONFIG_NAME} ==="
echo "Model dir: ${MODEL_DIR}"
echo "Test TSV: ${TEST_TSV}"
echo "Output dir: ${OUTPUT_DIR}"

run_shard() {
  local gpu_id="$1"
  local shard_id="$2"
  local suffix="$3"

  echo "=== Starting ${CONFIG_NAME} gated shard ${shard_id}/2 on GPU ${gpu_id} ==="
  CUDA_VISIBLE_DEVICES="${gpu_id}" python evaluate_whisper_validation.py \
    --eval_mode gated \
    --model_dir "${MODEL_DIR}" \
    --base_model openai/whisper-large-v3 \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --config_name "${CONFIG_NAME}" \
    --noise_condition "${NOISE_CONDITION}" \
    --noise_ratio "${NOISE_RATIO}" \
    --batch_size 4 \
    --audio_num_workers 4 \
    --lm_batch_size 8 \
    --shard_id "${shard_id}" \
    --num_shards 2 \
    --output_suffix "${suffix}" \
    --lm_models gpt2 Qwen/Qwen3-0.6B
}

run_shard 0 0 "_shard00-of-02" &
pid0=$!
run_shard 1 1 "_shard01-of-02" &
pid1=$!

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "ERROR: one or more gated shards failed for ${CONFIG_NAME} with status ${status}" >&2
  exit "${status}"
fi

echo "=== Done gated eval: ${CONFIG_NAME} ==="