#!/bin/bash
#SBATCH --job-name=hall_ctc_transfer
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_transfer.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_transfer.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
SOURCE_DIR=${ROOT}/hallucination_mitigation_acoustic_before_after
SCORED=${SOURCE_DIR}/scored_outputs.csv
GATE=${SOURCE_DIR}/frozen_gate_threshold.json
OUTDIR=${ROOT}/hallucination_mitigation_acoustic_clean_transfer

if [[ ! -f "${SCORED}" ]]; then
    echo "ERROR: missing scored outputs from first mitigation experiment: ${SCORED}" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"

cat <<EOF
=== Second mitigation experiment: clean-transfer risk/coverage ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Input scored outputs: ${SCORED}
Original frozen gate: ${GATE}
Output: ${OUTDIR}
Coverage sweep: ${COVERAGES:-0.99 0.98 0.95 0.90 0.85}
Protocol: thresholds calibrated on clean DEV only; TEST never retuned
EOF

python -u acoustic_abstention_clean_transfer.py \
    --scored_outputs "${SCORED}" \
    --frozen_gate_json "${GATE}" \
    --output_dir "${OUTDIR}" \
    --coverages ${COVERAGES:-0.99 0.98 0.95 0.90 0.85}

echo
echo "=== Done ==="
echo "Paper summary: ${OUTDIR}/paper_clean_transfer_summary.csv"
echo "Long results:  ${OUTDIR}/clean_transfer_by_condition.csv"
echo "Separability:  ${OUTDIR}/clean_transfer_separability.json"
echo "Report:        ${OUTDIR}/report.json"
