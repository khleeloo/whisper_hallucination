#!/bin/bash
#SBATCH --account=vemotionsys
#SBATCH --job-name=stress_ur64
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_ur64.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_ur64.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=normal

set -euo pipefail

source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
EVAL_PY="/home/rmfrieske/whisper_hallucination/evaluate_dual_metric.py"
TEST_TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
CONFIG="ur_64pct"
MODEL_DIR="${BASE_DIR}/ur_64pct/checkpoint-10000"
OUTDIR="${BASE_DIR}/stress_eval_64pct_ur_checkpoint-10000"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

mkdir -p "${OUTDIR}" /home/rmfrieske/whisper_hallucination/slurm_logs

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json: ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights: ${MODEL_DIR}" >&2
    exit 1
fi

run_eval() {
    local perturb_type="$1"
    local perturb_amp="$2"
    local perturb_dur="$3"
    local tag="${perturb_type}_amp${perturb_amp}_dur${perturb_dur}"
    local results_path="${OUTDIR}/results_${CONFIG}_${tag}.json"
    local details_path="${OUTDIR}/details_${CONFIG}_${tag}.tsv"

    if [[ -s "${results_path}" && -s "${details_path}" ]]; then
        echo "=== Skipping existing result: ${CONFIG} ${tag} ==="
        return 0
    fi

    rm -f "${results_path}" "${details_path}"

    python "${EVAL_PY}" \
        --model_dir "${MODEL_DIR}" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "${TEST_TSV}" \
        --clips_dir "${CLIPS}" \
        --output_dir "${OUTDIR}" \
        --config_name "${CONFIG}" \
        --max_samples "${MAX_SAMPLES}" \
        --batch_size 8 \
        --lm_models gpt2 \
        --perturb_type "${perturb_type}" \
        --perturb_amplitude "${perturb_amp}" \
        --perturb_duration "${perturb_dur}" \
        --num_log_examples 5
}

run_eval full_noise 0.5 0.0
run_eval full_noise 0.75 0.0
run_eval onset_noise 0.5 0.5
run_eval onset_noise 0.75 0.5
run_eval reverb 0.5 0.5
run_eval reverb 0.8 0.5

echo "UR 64% checkpoint-10000 missing stress evaluation complete: ${OUTDIR}"