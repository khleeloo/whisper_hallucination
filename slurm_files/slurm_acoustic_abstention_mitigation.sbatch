#!/bin/bash
#SBATCH --job-name=hall_ctc_gate
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_gate.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_gate.err

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
OUTDIR=${ROOT}/hallucination_mitigation_acoustic

mkdir -p "${OUTDIR}"

echo "=== Hallucination mitigation: acoustic-consistency abstention ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: ${MODEL_DIR}"
echo "Test TSV: ${TEST_TSV}"
echo "DEV TSV override: ${DEV_TSV:-auto-detect}"
echo "Output: ${OUTDIR}"
echo "DEV max samples: ${DEV_MAX_SAMPLES:-1000}"
echo "TEST max samples: ${TEST_MAX_SAMPLES:-1000}"
echo "Minimum clean DEV coverage: ${MIN_CLEAN_COVERAGE:-0.98}"
echo "Seed: ${SEED:-20260820}"

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json in ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights in ${MODEL_DIR}" >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ -n "${DEV_TSV:-}" ]]; then
    EXTRA_ARGS+=(--dev_tsv "${DEV_TSV}")
fi
if [[ "${REUSE_GENERATED:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--reuse_generated_outputs)
fi

python -u acoustic_abstention_mitigation.py \
    --output_dir "${OUTDIR}" \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --base_model_dir "${MODEL_DIR}" \
    --dev_max_samples "${DEV_MAX_SAMPLES:-1000}" \
    --test_max_samples "${TEST_MAX_SAMPLES:-1000}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --ctc_batch_size "${CTC_BATCH_SIZE:-8}" \
    --lm_batch_size "${LM_BATCH_SIZE:-8}" \
    --min_clean_coverage "${MIN_CLEAN_COVERAGE:-0.98}" \
    --hall_wer_threshold "${HALL_WER_THRESHOLD:-0.116215}" \
    --hall_plaus_threshold "${HALL_PLAUS_THRESHOLD:-0.892170}" \
    --bootstrap "${N_BOOT:-10000}" \
    --seed "${SEED:-20260820}" \
    --perturbations \
        none \
        full_noise_amp0.5_dur0.0 \
        full_noise_amp0.75_dur0.0 \
    "${EXTRA_ARGS[@]}"

echo
echo "=== Acoustic-consistency mitigation complete ==="
echo "Frozen threshold: ${OUTDIR}/frozen_gate_threshold.json"
echo "TEST summary:     ${OUTDIR}/test_mitigation_summary.csv"
echo "TEST rows:        ${OUTDIR}/test_outputs_with_gate.csv"
echo "Report:           ${OUTDIR}/report.json"
