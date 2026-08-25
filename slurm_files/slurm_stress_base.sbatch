#!/bin/bash
#SBATCH --job-name=acoustic_full
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_acoustic_full.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_acoustic_full.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
MODEL_DIR=${ROOT}/base/checkpoint-10000
TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
OUTDIR=${ROOT}/acoustic_stress_full

mkdir -p "${OUTDIR}"

echo "=== Canonical full acoustic-stress experiment ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: ${MODEL_DIR}"
echo "Test TSV: ${TEST_TSV}"
echo "Output: ${OUTDIR}"
echo "N per condition: ${MAX_SAMPLES:-1000}"
echo "LMs: Qwen/Qwen3-0.6B + GPT2"
echo "Seed: ${SEED:-20260821}"
echo
echo "Battery:"
echo "  clean"
echo "  silence"
echo "  leading silence: 1s, 3s"
echo "  onset Gaussian noise: 0.05, 0.5, 0.75"
echo "  reverb: 0.5, 0.8"
echo "  speech-band noise: 0.5, 0.75"
echo "  full-band Gaussian noise: 0.5, 0.75"

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json in ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights in ${MODEL_DIR}" >&2
    exit 1
fi
[[ -s "${TEST_TSV}" ]] || { echo "ERROR: missing test TSV: ${TEST_TSV}" >&2; exit 1; }
[[ -d "${CLIPS_DIR}" ]] || { echo "ERROR: missing clips dir: ${CLIPS_DIR}" >&2; exit 1; }

EXTRA_ARGS=()
if [[ "${REUSE_GENERATED:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--reuse_generated_outputs)
fi

python -u acoustic_stress_experiment.py \
    --output_dir "${OUTDIR}" \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --base_model_dir "${MODEL_DIR}" \
    --qwen_model Qwen/Qwen3-0.6B \
    --gpt2_model gpt2 \
    --max_samples "${MAX_SAMPLES:-1000}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --lm_batch_size "${LM_BATCH_SIZE:-8}" \
    --bootstrap "${N_BOOT:-10000}" \
    --seed "${SEED:-20260821}" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== Full acoustic-stress experiment complete ==="
echo "Per utterance: ${OUTDIR}/per_utterance_acoustic_stress.csv"
echo "Summary:       ${OUTDIR}/acoustic_stress_summary.csv"
echo "Top outputs:   ${OUTDIR}/top_hypotheses_by_condition.csv"
echo "Thresholds:    ${OUTDIR}/matched_clean_dual_lm_thresholds.json"
echo "Report:        ${OUTDIR}/report.json"
