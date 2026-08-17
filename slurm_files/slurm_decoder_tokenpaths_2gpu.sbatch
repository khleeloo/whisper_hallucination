#!/bin/bash
#SBATCH --job-name=dec_tokp2
#SBATCH --partition=normal
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_decoder_tokenpaths_2gpu.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_decoder_tokenpaths_2gpu.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
TEST_TSV_64="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
TEST_TSV_100="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct/test.tsv"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/stress_eval_full_noise_decoder_tokenpaths_2gpu_b8}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_HYPOTHESES="${N_HYPOTHESES:-3}"
N_BEAMS="${N_BEAMS:-3}"

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

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ "${#CUDA_DEVICE_LIST[@]}" -lt 2 ]]; then
  echo "ERROR: expected two visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

run_task() {
  local gpu_slot="$1"
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
  local task_log="/home/rmfrieske/whisper_hallucination/slurm_logs/${SLURM_JOB_ID}_${config_name}_${tag}_gpu${gpu_slot}.log"

  if [[ -s "${result_json}" && -s "${detail_tsv}" ]]; then
    echo "Skipping existing decoder token-path result: ${config_name} ${tag}"
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

  {
    echo "============================================================"
    echo "Decoder token-path full-noise eval"
    echo "Config: ${config_name}"
    echo "Model: ${model_dir}"
    echo "Test TSV: ${test_tsv}"
    echo "Perturbation: ${tag}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Max samples: ${MAX_SAMPLES}"
    echo "Batch/token paths/beams: ${BATCH_SIZE}/${N_HYPOTHESES}/${N_BEAMS}"
    echo "GPU slot/device: ${gpu_slot}/${CUDA_DEVICE_LIST[$gpu_slot]}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST[$gpu_slot]}" python evaluate_dual_metric.py \
      --eval_mode gated \
      --model_dir "${model_dir}" \
      --base_model openai/whisper-large-v3 \
      --test_tsv "${test_tsv}" \
      --clips_dir "${CLIPS_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --config_name "${config_name}" \
      --max_samples "${MAX_SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --lm_models gpt2 \
      --perturb_type full_noise \
      --perturb_amplitude "${amp}" \
      --perturb_duration 0.0 \
      --num_decoder_hypotheses "${N_HYPOTHESES}" \
      --decoder_hypothesis_beams "${N_BEAMS}" \
      --gate_token_path_distance_threshold 0.25 \
      --gate_beam_logprob_margin_threshold 0.25 \
      --gate_speech_fraction_threshold -1 \
      --gate_snr_proxy_threshold -999 \
      --num_log_examples 5

    echo "Done: ${config_name} ${tag}"
  } 2>&1 | tee "${task_log}"
}

echo "Running ${N_TASKS} decoder token-path full-noise tasks, two at a time on two GPUs."
echo "Output: ${OUTPUT_DIR}"
echo "Visible GPU list: ${CUDA_DEVICE_LIST[*]}"

for ((task_id = 0; task_id < N_TASKS; task_id += 2)); do
  run_task 0 "${task_id}" &
  pid0=$!

  pid1=""
  if (( task_id + 1 < N_TASKS )); then
    run_task 1 "$((task_id + 1))" &
    pid1=$!
  fi

  wait "${pid0}"
  if [[ -n "${pid1}" ]]; then
    wait "${pid1}"
  fi
done

echo "All decoder token-path full-noise evaluations complete: ${OUTPUT_DIR}"