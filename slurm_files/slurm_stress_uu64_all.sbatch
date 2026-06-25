#!/bin/bash
#SBATCH --account=vemotionsys
#SBATCH --job-name=stress_uu64
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_uu64.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_uu64.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=normal

set -euo pipefail

source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
EVAL_PY="/home/rmfrieske/whisper_hallucination/evaluate_dual_metric.py"
TEST_TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
CONFIG="uu_64pct"
MODEL_DIR="${BASE_DIR}/uu_64pct/checkpoint-9375"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json for ${CONFIG}: ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights for ${CONFIG}: ${MODEL_DIR}" >&2
    exit 1
fi

run_eval() {
    local outdir="$1"
    local perturb_type="$2"
    local perturb_amp="$3"
    local perturb_dur="$4"
    local tag="${perturb_type}_amp${perturb_amp}_dur${perturb_dur}"
    local results_path="${outdir}/results_${CONFIG}_${tag}.json"
    local details_path="${outdir}/details_${CONFIG}_${tag}.tsv"

    mkdir -p "${outdir}" /home/rmfrieske/whisper_hallucination/slurm_logs

    if [[ -s "${results_path}" && -s "${details_path}" ]]; then
        echo "=== Skipping existing result: ${CONFIG} ${tag} ==="
        return 0
    fi

    rm -f "${results_path}" "${details_path}"

    echo "=== Stress eval ==="
    echo "Config: ${CONFIG}"
    echo "Model: ${MODEL_DIR}"
    echo "Perturbation: ${perturb_type}, amp=${perturb_amp}, dur=${perturb_dur}"
    echo "Perturbation tag: ${tag}"
    echo "Max samples: ${MAX_SAMPLES}"
    echo "Output: ${outdir}"

    python "${EVAL_PY}" \
        --model_dir "${MODEL_DIR}" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "${TEST_TSV}" \
        --clips_dir "${CLIPS}" \
        --output_dir "${outdir}" \
        --config_name "${CONFIG}" \
        --max_samples "${MAX_SAMPLES}" \
        --batch_size 8 \
        --lm_models gpt2 \
        --perturb_type "${perturb_type}" \
        --perturb_amplitude "${perturb_amp}" \
        --perturb_duration "${perturb_dur}" \
        --num_log_examples 5
}

ORIG_OUTDIR="${BASE_DIR}/stress_eval_64pct"
NEW_OUTDIR="${BASE_DIR}/stress_eval_new_conditions_64pct"

# Original stress table conditions.
run_eval "${ORIG_OUTDIR}" full_noise 0.5 0.0
run_eval "${ORIG_OUTDIR}" full_noise 0.75 0.0
run_eval "${ORIG_OUTDIR}" onset_noise 0.5 0.5
run_eval "${ORIG_OUTDIR}" onset_noise 0.75 0.5
run_eval "${ORIG_OUTDIR}" reverb 0.5 0.5
run_eval "${ORIG_OUTDIR}" reverb 0.8 0.5

# New stress table conditions.
run_eval "${NEW_OUTDIR}" silence 0.0 0.0
run_eval "${NEW_OUTDIR}" leading_silence 0.0 1.0
run_eval "${NEW_OUTDIR}" leading_silence 0.0 3.0
run_eval "${NEW_OUTDIR}" speech_band_noise 0.5 0.0
run_eval "${NEW_OUTDIR}" speech_band_noise 0.75 0.0

echo "UU 64% stress evaluation complete."
