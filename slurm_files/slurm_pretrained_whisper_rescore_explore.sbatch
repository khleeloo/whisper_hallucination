#!/bin/bash
#SBATCH --job-name=whisper_rescore
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_whisper_rescore.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_whisper_rescore.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
SOURCE=${ROOT}/pretrained_whisper_stress_pipeline/scored_outputs.csv
OUTDIR=${ROOT}/pretrained_whisper_stress_pipeline/rescore_explore

if [[ ! -f "${SOURCE}" ]]; then
    echo "ERROR: missing source ${SOURCE}" >&2
    exit 1
fi

cat <<EOF
=== Corrected WER rescore + pretrained Whisper failure exploration ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Source: ${SOURCE}
Output: ${OUTDIR}
Strict WER: ${STRICT_WER:-0.5}
Min clean DEV coverage: ${MIN_CLEAN_COVERAGE:-0.95}
Top hypotheses/condition: ${TOP_K:-20}
EOF

python -u pretrained_whisper_rescore_explore.py \
    --source "${SOURCE}" \
    --output_dir "${OUTDIR}" \
    --strict_wer "${STRICT_WER:-0.5}" \
    --min_clean_coverage "${MIN_CLEAN_COVERAGE:-0.95}" \
    --top_k "${TOP_K:-20}"

echo
echo "=== Done ==="
echo "Report:          ${OUTDIR}/rescore_report.json"
echo "Corrected TEST:  ${OUTDIR}/test_condition_summary_corrected.csv"
echo "Corrected gate:  ${OUTDIR}/test_gate_summary_corrected.csv"
echo "Clean candidates:${OUTDIR}/strict_clean_hallucination_candidates_corrected.csv"
echo "Top hypotheses:  ${OUTDIR}/top_hypotheses_by_condition.csv"
echo "Changed rows:    ${OUTDIR}/rows_changed_by_wer_fix.csv"
