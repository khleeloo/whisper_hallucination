#!/bin/bash
#SBATCH --job-name=wh64_eval
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_64pct_eval.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_64pct_eval.err
#SBATCH --mem 64G
#SBATCH --array=0-1  # 0=rr, 1=ru

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
TEST_TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/rmfrieske/whisper_hallucination/slurm_logs

CONFIGS=("rr_64pct" "ru_64pct")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONDITION="${CONFIG%%_*}"
MODEL_DIR="${BASE_DIR}/${CONFIG}"

# Find final adapter first, then newest numeric adapter checkpoint.
if [ -d "${MODEL_DIR}/final" ]; then
    CKPT_DIR="${MODEL_DIR}/final"
    CKPT_NAME="final"
else
    CKPT_DIR=""
    while IFS= read -r candidate; do
        if [ -f "${candidate}/adapter_config.json" ] && \
           { [ -f "${candidate}/adapter_model.safetensors" ] || [ -f "${candidate}/adapter_model.bin" ]; }; then
            CKPT_DIR="${candidate}"
            break
        fi
    done < <(find "${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' 2>/dev/null | sort -V -r)
    if [ -z "${CKPT_DIR}" ]; then
        echo "ERROR: No complete adapter checkpoint found in ${MODEL_DIR}"
        exit 1
    fi
    CKPT_NAME=$(basename "${CKPT_DIR}")
fi

echo "=== Evaluating ${CONFIG}/${CKPT_NAME} ==="
echo "Model dir: ${CKPT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

run_shard() {
    local gpu_id="$1"
    local shard_id="$2"
    local suffix="$3"

    echo "=== Starting ${CONFIG}/${CKPT_NAME} shard ${shard_id}/2 on GPU ${gpu_id} ==="
    CUDA_VISIBLE_DEVICES="${gpu_id}" python evaluate_whisper_validation.py \
        --model_dir "${CKPT_DIR}" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "${TEST_TSV}" \
        --clips_dir "${CLIPS_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --config_name "${CONFIG}_${CKPT_NAME}" \
        --noise_condition "${CONDITION}" \
        --noise_ratio 0.64 \
        --batch_size 8 \
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
if [ "${status}" -ne 0 ]; then
    echo "ERROR: one or more shards failed with status ${status}" >&2
    exit "${status}"
fi

echo "=== Done: ${CONFIG}/${CKPT_NAME} ==="
