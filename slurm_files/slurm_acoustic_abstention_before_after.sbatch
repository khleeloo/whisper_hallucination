#!/bin/bash
#SBATCH --job-name=hall_ctc_ba
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_ba.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_ba.err

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
OUTDIR=${ROOT}/hallucination_mitigation_acoustic_before_after

mkdir -p "${OUTDIR}"

EXTRA_ARGS=()
if [[ -n "${DEV_TSV:-}" ]]; then
    EXTRA_ARGS+=(--dev_tsv "${DEV_TSV}")
fi
if [[ "${REUSE_GENERATED:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--reuse_generated_outputs)
fi
if [[ -n "${WAV2VEC2_MODEL:-}" ]]; then
    EXTRA_ARGS+=(--wav2vec2_model "${WAV2VEC2_MODEL}")
fi

cat <<EOF
=== Paper experiment: acoustic-consistency abstention before/after ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Model: ${MODEL_DIR}
DEV TSV: ${DEV_TSV:-auto-detect}
TEST TSV: ${TEST_TSV}
Output: ${OUTDIR}
Protocol: clean + full-noise 0.50 + full-noise 0.75
Gate tuning: DEV only, Qwen3-primary objective
Clean DEV coverage constraint: ${MIN_CLEAN_COVERAGE:-0.98}
TEST: frozen gate, no retuning
DEV samples/condition: ${DEV_MAX_SAMPLES:-1000}
TEST samples/condition: ${TEST_MAX_SAMPLES:-1000}
Bootstrap: ${N_BOOT:-10000}
Seed: ${SEED:-20260820}
EOF

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json in ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights in ${MODEL_DIR}" >&2
    exit 1
fi

python -u acoustic_abstention_before_after.py \
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
    --bootstrap "${N_BOOT:-10000}" \
    --seed "${SEED:-20260820}" \
    --qwen_model "${QWEN_MODEL:-Qwen/Qwen3-0.6B}" \
    --gpt2_model "${GPT2_MODEL:-gpt2}" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== Done ==="
echo "Paper table:       ${OUTDIR}/paper_before_after_summary.csv"
echo "Detailed TEST:     ${OUTDIR}/test_mitigation_summary.csv"
echo "Per-output TEST:   ${OUTDIR}/test_outputs_with_gate.csv"
echo "Frozen gate:       ${OUTDIR}/frozen_gate_threshold.json"
echo "Hallucination thr: ${OUTDIR}/frozen_hallucination_thresholds_dual_lm.json"
echo "Report:            ${OUTDIR}/report.json"
