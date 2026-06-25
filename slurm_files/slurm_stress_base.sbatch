#!/bin/bash
#SBATCH --account=vemotionsys
#SBATCH --job-name=stress_base
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_base.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_base.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=normal

set -euo pipefail

source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
EVAL_PY="/home/rmfrieske/whisper_hallucination/evaluate_dual_metric.py"
TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTDIR="${BASE_DIR}/stress_eval_64pct"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

CONFIG="base"
MODEL_DIR="${BASE_DIR}/base/checkpoint-10000"

PERTURB_TYPES=("full_noise" "full_noise" "onset_noise" "onset_noise" "reverb" "reverb")
PERTURB_AMPS=("0.5" "0.75" "0.5" "0.75" "0.5" "0.8")
PERTURB_DURS=("0.0" "0.0" "0.5" "0.5" "0.5" "0.5")

mkdir -p "${OUTDIR}" /home/rmfrieske/whisper_hallucination/slurm_logs

if [ ! -f "${MODEL_DIR}/adapter_config.json" ]; then
    echo "ERROR: missing adapter_config.json for ${CONFIG}: ${MODEL_DIR}" >&2
    exit 1
fi
if [ ! -f "${MODEL_DIR}/adapter_model.safetensors" ] && [ ! -f "${MODEL_DIR}/adapter_model.bin" ]; then
    echo "ERROR: missing adapter weights for ${CONFIG}: ${MODEL_DIR}" >&2
    exit 1
fi

for IDX in "${!PERTURB_TYPES[@]}"; do
    PERTURB_TYPE="${PERTURB_TYPES[$IDX]}"
    PERTURB_AMP="${PERTURB_AMPS[$IDX]}"
    PERTURB_DUR="${PERTURB_DURS[$IDX]}"
    PERTURB_TAG="${PERTURB_TYPE}_amp${PERTURB_AMP}_dur${PERTURB_DUR}"
    RESULTS_PATH="${OUTDIR}/results_${CONFIG}_${PERTURB_TAG}.json"
    DETAILS_PATH="${OUTDIR}/details_${CONFIG}_${PERTURB_TAG}.tsv"

    if [ -s "${RESULTS_PATH}" ] && [ -s "${DETAILS_PATH}" ]; then
        echo "=== Skipping existing result: ${CONFIG} ${PERTURB_TAG} ==="
        continue
    fi

    rm -f "${RESULTS_PATH}" "${DETAILS_PATH}"

    echo "=== Base stress eval ==="
    echo "Config: ${CONFIG}"
    echo "Model: ${MODEL_DIR}"
    echo "Perturbation: ${PERTURB_TYPE}, amp=${PERTURB_AMP}, dur=${PERTURB_DUR}"
    echo "Perturbation tag: ${PERTURB_TAG}"
    echo "Max samples: ${MAX_SAMPLES}"
    echo "Output: ${OUTDIR}"

    python "${EVAL_PY}" \
        --model_dir "${MODEL_DIR}" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "${TSV}" \
        --clips_dir "${CLIPS}" \
        --output_dir "${OUTDIR}" \
        --config_name "${CONFIG}" \
        --max_samples "${MAX_SAMPLES}" \
        --batch_size 8 \
        --lm_models gpt2 \
        --perturb_type "${PERTURB_TYPE}" \
        --perturb_amplitude "${PERTURB_AMP}" \
        --perturb_duration "${PERTURB_DUR}" \
        --num_log_examples 5
done