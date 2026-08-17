#!/bin/bash
#SBATCH --job-name=gate_full_noise
#SBATCH --partition=normal
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=192G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_gate_full_noise.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_gate_full_noise.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
TEST_TSV_64="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
TEST_TSV_100="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct/test.tsv"
OUTPUT_DIR="${BASE_DIR}/stress_eval_full_noise_gated_acoustic_all"
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
N_MODELS="${#CONFIG_NAMES[@]}"
N_TASKS=$((N_MODELS * ${#AMPS[@]}))
N_GPUS=4

run_task() {
  local gpu_id="$1"
  local task_id="$2"
  local model_idx=$((task_id / ${#AMPS[@]}))
  local amp_idx=$((task_id % ${#AMPS[@]}))
  local config_name="${CONFIG_NAMES[$model_idx]}"
  local model_dir="${MODEL_DIRS[$model_idx]}"
  local test_tsv="${TEST_TSVS[$model_idx]}"
  local amp="${AMPS[$amp_idx]}"
  local tag="full_noise_amp${amp}_dur0.0"
  local result_json="${OUTPUT_DIR}/results_${config_name}_${tag}_gated.json"
  local detail_tsv="${OUTPUT_DIR}/details_${config_name}_${tag}_gated.tsv"

  if [[ -s "${result_json}" && -s "${detail_tsv}" ]]; then
    echo "[gpu ${gpu_id}] Skipping existing gated full-noise result: ${config_name} ${tag}"
    return 0
  fi

  if [[ ! -f "${model_dir}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json in ${model_dir}" >&2
    return 1
  fi
  if [[ ! -f "${model_dir}/adapter_model.safetensors" && ! -f "${model_dir}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights in ${model_dir}" >&2
    return 1
  fi

  echo "============================================================"
  echo "[gpu ${gpu_id}] Gated full-noise eval"
  echo "Config: ${config_name}"
  echo "Model: ${model_dir}"
  echo "Test TSV: ${test_tsv}"
  echo "Perturbation: ${tag}"
  echo "Output: ${OUTPUT_DIR}"
  echo "Max samples: ${MAX_SAMPLES}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python evaluate_dual_metric.py \
    --eval_mode gated \
    --model_dir "${model_dir}" \
    --base_model openai/whisper-large-v3 \
    --test_tsv "${test_tsv}" \
    --clips_dir "${CLIPS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --config_name "${config_name}" \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size 4 \
    --lm_models gpt2 \
    --perturb_type full_noise \
    --perturb_amplitude "${amp}" \
    --perturb_duration 0.0 \
    --num_log_examples 5

  echo "[gpu ${gpu_id}] Done: ${config_name} ${tag}"
}

worker() {
  local gpu_id="$1"
  local task_id
  for ((task_id = gpu_id; task_id < N_TASKS; task_id += N_GPUS)); do
    run_task "${gpu_id}" "${task_id}"
  done
}

echo "Running ${N_TASKS} gated full-noise tasks across ${N_GPUS} GPUs."

pids=()
for gpu_id in 0 1 2 3; do
  worker "${gpu_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done

if [[ "${status}" -ne 0 ]]; then
  echo "ERROR: one or more gated full-noise workers failed with status ${status}" >&2
  exit "${status}"
fi

echo "All gated full-noise evaluations complete: ${OUTPUT_DIR}"