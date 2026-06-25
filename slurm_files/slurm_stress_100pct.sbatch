#!/bin/bash
#SBATCH --job-name=stress100
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress100.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress100.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
EVAL_PY="/home/rmfrieske/whisper_hallucination/evaluate_dual_metric.py"
TEST_TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct/test.tsv"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTDIR="${BASE_DIR}/stress_eval_100pct"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

mkdir -p "${OUTDIR}" /home/rmfrieske/whisper_hallucination/slurm_logs

CONFIGS=(uu_100pct ur_100pct ru_100pct rr_100pct)
PERTURB_TYPES=("full_noise" "full_noise" "onset_noise" "onset_noise" "reverb" "reverb")
PERTURB_AMPS=("0.5" "0.75" "0.5" "0.75" "0.5" "0.8")
PERTURB_DURS=("0.0" "0.0" "0.5" "0.5" "0.5" "0.5")

if [ -n "${STRESS_CONFIG:-}" ]; then
    CONFIGS=("${STRESS_CONFIG}")
fi

resolve_model_dir() {
    local config="$1"
    local run_dir="${BASE_DIR}/${config}"

    if [ -d "${run_dir}/final" ]; then
        printf '%s\n' "${run_dir}/final"
        return 0
    fi

    while IFS= read -r candidate; do
        if [ -f "${candidate}/adapter_config.json" ] && \
           { [ -f "${candidate}/adapter_model.safetensors" ] || [ -f "${candidate}/adapter_model.bin" ]; }; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done < <(find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' 2>/dev/null | sort -V -r)

    return 1
}

for CONFIG in "${CONFIGS[@]}"; do
    MODEL_DIR="$(resolve_model_dir "${CONFIG}")" || {
        echo "ERROR: no final or complete checkpoint found for ${CONFIG}" >&2
        exit 1
    }

    echo "============================================================"
    echo "Config: ${CONFIG}"
    echo "Model:  ${MODEL_DIR}"
    echo "Output: ${OUTDIR}"
    echo "Max samples: ${MAX_SAMPLES}"
    echo "============================================================"

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

        echo "=== Stress eval ==="
        echo "Config: ${CONFIG}"
        echo "Perturbation: ${PERTURB_TYPE}, amp=${PERTURB_AMP}, dur=${PERTURB_DUR}"

        python "${EVAL_PY}" \
            --model_dir "${MODEL_DIR}" \
            --base_model openai/whisper-large-v3 \
            --test_tsv "${TEST_TSV}" \
            --clips_dir "${CLIPS_DIR}" \
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
done

echo "100pct stress evaluation complete: ${OUTDIR}"