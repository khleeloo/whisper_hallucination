#!/bin/bash
#SBATCH --job-name=wh100_eval
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_100pct_eval.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_100pct_eval.err
#SBATCH --mem 128G

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="${BASE_DIR}/eval_100pct"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/rmfrieske/whisper_hallucination/slurm_logs

CONFIGS=(uu ur ru rr)
NOISE_CONDITIONS=(UU UR RU RR)

for index in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$index]}"
    CONDITION="${NOISE_CONDITIONS[$index]}"
    RUN_DIR="${BASE_DIR}/${CONFIG}_100pct"
    MODEL_DIR="${RUN_DIR}/final"

    if [ ! -d "${MODEL_DIR}" ]; then
        echo "Final model missing for ${CONFIG}_100pct; searching latest complete checkpoint..."
        MODEL_DIR=""
        while IFS= read -r candidate; do
            if [ -f "${candidate}/adapter_config.json" ] && \
               { [ -f "${candidate}/adapter_model.safetensors" ] || [ -f "${candidate}/adapter_model.bin" ]; }; then
                MODEL_DIR="${candidate}"
                break
            fi
        done < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' 2>/dev/null | sort -V -r)
    fi

    if [ -z "${MODEL_DIR}" ] || [ ! -d "${MODEL_DIR}" ]; then
        echo "ERROR: No final or complete checkpoint found for ${CONFIG}_100pct in ${RUN_DIR}" >&2
        exit 1
    fi

    echo "=== Evaluating ${CONFIG}_100pct (${CONDITION}) ==="
    echo "Model dir: ${MODEL_DIR}"
    echo "Output dir: ${OUTPUT_DIR}"

    python evaluate_whisper_validation.py \
        --model_dir "${MODEL_DIR}" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "${DATA_DIR}/test.tsv" \
        --clips_dir "${CLIPS_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --config_name "${CONFIG}_100pct" \
        --noise_condition "${CONDITION}" \
        --noise_ratio 1.0 \
        --batch_size 8 \
        --lm_batch_size 8 \
        --lm_models gpt2 Qwen/Qwen3-0.6B

    echo "=== Eval complete for ${CONFIG}_100pct ==="
done

echo "=== All 100pct evals complete ==="