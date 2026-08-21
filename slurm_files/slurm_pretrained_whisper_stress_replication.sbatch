#!/bin/bash
#SBATCH --job-name=whisper_pre_stress
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_whisper_pre_stress.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_whisper_pre_stress.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
OUTDIR=${ROOT}/pretrained_whisper_stress_pipeline
TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips

EXTRA_ARGS=()
if [[ -n "${DEV_TSV:-}" ]]; then
    EXTRA_ARGS+=(--dev_tsv "${DEV_TSV}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--resume)
fi
if [[ -n "${WAV2VEC2_MODEL:-}" ]]; then
    EXTRA_ARGS+=(--wav2vec2_model "${WAV2VEC2_MODEL}")
fi

cat <<EOF
=== Untouched pretrained Whisper-large-v3 replication ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Whisper model: ${WHISPER_MODEL:-openai/whisper-large-v3}
Protocol: clean + full-noise 0.50 + full-noise 0.75
DEV TSV: ${DEV_TSV:-auto-detect}
TEST TSV: ${TEST_TSV}
Output: ${OUTDIR}
Corrected WER: yes
Diagnostic H: WER > clean-DEV mean + LM > clean-DEV mean
Strict H: WER > ${STRICT_WER:-0.5} + LM > clean-DEV mean
Gate calibration: stressed DEV strict-Qwen failures, subject to clean DEV coverage >= ${MIN_CLEAN_COVERAGE:-0.95}
Gate TEST: frozen, reference-free, no retuning
DEV samples/condition: ${DEV_MAX_SAMPLES:-1000}
TEST samples/condition: ${TEST_MAX_SAMPLES:-1000}
Seed: ${SEED:-20260821}
EOF

python -u pretrained_whisper_stress_replication.py \
    --output_dir "${OUTDIR}" \
    --whisper_model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --dev_max_samples "${DEV_MAX_SAMPLES:-1000}" \
    --test_max_samples "${TEST_MAX_SAMPLES:-1000}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --lm_batch_size "${LM_BATCH_SIZE:-8}" \
    --ctc_batch_size "${CTC_BATCH_SIZE:-8}" \
    --strict_wer "${STRICT_WER:-0.5}" \
    --min_clean_coverage "${MIN_CLEAN_COVERAGE:-0.95}" \
    --seed "${SEED:-20260821}" \
    --qwen_model "${QWEN_MODEL:-Qwen/Qwen3-0.6B}" \
    --gpt2_model "${GPT2_MODEL:-gpt2}" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== Done ==="
echo "Headline:        ${OUTDIR}/headline_summary.txt"
echo "TEST stress:     ${OUTDIR}/test_condition_summary.csv"
echo "TEST gate:       ${OUTDIR}/test_gate_summary.csv"
echo "Strict clean H:  ${OUTDIR}/strict_clean_hallucination_candidates.csv"
echo "Frozen gate:     ${OUTDIR}/frozen_gate_threshold.json"
echo "Thresholds:      ${OUTDIR}/frozen_hallucination_thresholds.json"
echo "Per utterance:   ${OUTDIR}/scored_outputs.csv"
echo "Report:          ${OUTDIR}/report.json"
