#!/bin/bash
#SBATCH --job-name=gate_full_noise
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --array=0-13
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_gate_full_noise.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_gate_full_noise.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
TEST_TSV_64="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
TEST_TSV_100="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct/test.tsv"
OUTPUT_DIR="${BASE_DIR}/stress_eval_full_noise_gated_all"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

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

TEST_TSVS=(
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_64}"
  "${TEST_TSV_100}"
  "${TEST_TSV_100}"
)

AMPS=("0.5" "0.75")

task_id="${SLURM_ARRAY_TASK_ID}"
model_idx=$((task_id / 2))
amp_idx=$((task_id % 2))

CONFIG_NAME="${CONFIG_NAMES[$model_idx]}"
MODEL_DIR="${MODEL_DIRS[$model_idx]}"
TEST_TSV="${TEST_TSVS[$model_idx]}"
AMP="${AMPS[$amp_idx]}"

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
  echo "ERROR: missing adapter_config.json in ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
  echo "ERROR: missing adapter weights in ${MODEL_DIR}" >&2
  exit 1
fi

TAG="full_noise_amp${AMP}_dur0.0"
RESULT_JSON="${OUTPUT_DIR}/results_${CONFIG_NAME}_${TAG}_gated.json"
DETAIL_TSV="${OUTPUT_DIR}/details_${CONFIG_NAME}_${TAG}_gated.tsv"

if [[ -s "${RESULT_JSON}" && -s "${DETAIL_TSV}" ]]; then
  echo "Skipping existing gated full-noise result: ${CONFIG_NAME} ${TAG}"
  exit 0
fi

echo "=== Gated full-noise eval ==="
echo "Config: ${CONFIG_NAME}"
echo "Model: ${MODEL_DIR}"
echo "Test TSV: ${TEST_TSV}"
echo "Perturbation: ${TAG}"
echo "Output: ${OUTPUT_DIR}"
echo "Max samples: ${MAX_SAMPLES}"

python evaluate_dual_metric.py \
  --eval_mode gated \
  --model_dir "${MODEL_DIR}" \
  --base_model openai/whisper-large-v3 \
  --test_tsv "${TEST_TSV}" \
  --clips_dir "${CLIPS_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_name "${CONFIG_NAME}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size 4 \
  --lm_models gpt2 \
  --perturb_type full_noise \
  --perturb_amplitude "${AMP}" \
  --perturb_duration 0.0 \
  --num_log_examples 5

echo "=== Done gated full-noise eval: ${CONFIG_NAME} ${TAG} ==="