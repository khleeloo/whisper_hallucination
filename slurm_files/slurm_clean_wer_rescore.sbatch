#!/bin/bash
#SBATCH --job-name=clean_wer_rescore
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_clean_wer_rescore.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_clean_wer_rescore.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
SOURCE=${SOURCE:-${ROOT}/hallucination_mitigation_acoustic_before_after/scored_outputs.csv}
STRESS_SOURCE=${STRESS_SOURCE:-${ROOT}/acoustic_stress_full/per_utterance_acoustic_stress.csv}
OUTDIR=${OUTDIR:-${ROOT}/clean_wer_rescore}

if [[ ! -f "${SOURCE}" ]]; then
    echo "ERROR: source scored_outputs.csv not found: ${SOURCE}" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"

cat <<EOF
=== Clean WER + hallucination rescoring ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Source: ${SOURCE}
Stress source: ${STRESS_SOURCE}
Output: ${OUTDIR}
Method: cached Whisper/Qwen/GPT2/CTC scores; recompute WER + thresholds + H labels
EOF

python -u clean_wer_rescore.py \
    --source "${SOURCE}" \
    --stress_source "${STRESS_SOURCE}" \
    --output_dir "${OUTDIR}"

echo
echo "=== Done ==="
echo "Normalization report: ${OUTDIR}/normalization_report.json"
echo "Corrected scores:     ${OUTDIR}/scored_outputs_cleanwer.csv"
echo "TEST mitigation:      ${OUTDIR}/test_mitigation_summary_cleanwer.csv"
echo "Risk/coverage:        ${OUTDIR}/clean_transfer_risk_coverage_cleanwer.csv"
echo "Clean H union:        ${OUTDIR}/clean_test_hallucinations_union_cleanwer.csv"
echo "Stress summary:       ${OUTDIR}/acoustic_stress_summary_cleanwer.csv"
