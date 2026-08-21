#!/bin/bash
#SBATCH --job-name=hall_clean_split
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_clean_split.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_clean_split.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
SCORED=${ROOT}/hallucination_mitigation_acoustic_before_after/scored_outputs.csv
OUTDIR=${ROOT}/clean_hallucinations_best_gate
TARGET_COV=${TARGET_CLEAN_DEV_COVERAGE:-0.95}

if [[ ! -f "${SCORED}" ]]; then
    echo "ERROR: missing scored outputs: ${SCORED}" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"

cat <<EOF
=== Clean hallucination separation: before vs after selected gate ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Scored outputs: ${SCORED}
Target clean DEV coverage: ${TARGET_COV}
Output: ${OUTDIR}
EOF

python -u clean_hallucination_gate_analysis.py \
    --scored_outputs "${SCORED}" \
    --output_dir "${OUTDIR}" \
    --target_clean_dev_coverage "${TARGET_COV}"

echo
echo "=== Done ==="
echo "All hallucinations union: ${OUTDIR}/all_clean_test_hallucinations_union.csv"
echo "Summary:                ${OUTDIR}/before_after_hallucination_summary.csv"
echo "Qwen before:            ${OUTDIR}/qwen_hallucinations_before.csv"
echo "Qwen caught:            ${OUTDIR}/qwen_hallucinations_caught.csv"
echo "Qwen missed after:      ${OUTDIR}/qwen_hallucinations_after_missed.csv"
echo "GPT2 before:            ${OUTDIR}/gpt2_hallucinations_before.csv"
echo "GPT2 caught:            ${OUTDIR}/gpt2_hallucinations_caught.csv"
echo "GPT2 missed after:      ${OUTDIR}/gpt2_hallucinations_after_missed.csv"
echo "Report:                 ${OUTDIR}/report.json"
