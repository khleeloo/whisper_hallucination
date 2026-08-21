#!/bin/bash
#SBATCH --job-name=hall_ctc_transfer_ext
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_transfer_ext.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_hall_ctc_transfer_ext.err

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
OUTDIR=${ROOT}/hallucination_mitigation_acoustic_clean_transfer_extended

# Preserve the completed paper-facing sweep and extend it downward to expose
# the risk/coverage knee. Thresholds are still calibrated on clean DEV only
# and then frozen before TEST evaluation.
DEFAULT_COVERAGES="0.99 0.98 0.95 0.90 0.85 0.80 0.75 0.70 0.60"
COVERAGE_SWEEP=${COVERAGES:-${DEFAULT_COVERAGES}}

if [[ ! -f "${SCORED}" ]]; then
    echo "ERROR: missing scored outputs from first mitigation experiment: ${SCORED}" >&2
    exit 1
fi

mkdir -p "${OUTDIR}"

cat <<EOF
=== Extended clean-transfer risk/coverage experiment ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Input scored outputs: ${SCORED}
Original frozen gate: ${GATE}
Output: ${OUTDIR}
Coverage sweep: ${COVERAGE_SWEEP}
Protocol: thresholds calibrated on clean DEV only; TEST never retuned
Purpose: extend below 85% DEV target coverage to locate the clean risk/coverage knee
EOF

python -u acoustic_abstention_clean_transfer.py \
    --scored_outputs "${SCORED}" \
    --frozen_gate_json "${GATE}" \
    --output_dir "${OUTDIR}" \
    --coverages ${COVERAGE_SWEEP}

# Fail closed if the extension did not actually produce all requested lower
# operating points. Also report whether severe-noise hallucination capture is
# still 100% at every point, without making that an assumption in the code.
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
out = root / "hallucination_mitigation_acoustic_clean_transfer_extended"
summary = pd.read_csv(out / "paper_clean_transfer_summary.csv")

required = {
    "clean_cov_0.800",
    "clean_cov_0.750",
    "clean_cov_0.700",
    "clean_cov_0.600",
}
present = set(summary["operating_point"].astype(str))
missing = sorted(required - present)
if missing:
    raise SystemExit(f"ERROR: missing extended operating points: {missing}")

cols = [
    "operating_point",
    "threshold",
    "clean_coverage_qwen",
    "clean_H_among_emitted_qwen",
    "clean_H_capture_qwen",
    "clean_rejection_precision_qwen",
    "full05_H_capture_qwen",
    "full075_H_capture_qwen",
]
print("\n=== Extended Qwen clean risk/coverage summary ===")
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

stress_cols = ["full05_H_capture_qwen", "full075_H_capture_qwen"]
all_full_noise_captured = bool((summary[stress_cols].fillna(0.0) >= 1.0 - 1e-12).all().all())
print(f"\n100% severe-noise hallucination capture at every operating point: {all_full_noise_captured}")
PY

echo
echo "=== Done ==="
echo "Paper summary: ${OUTDIR}/paper_clean_transfer_summary.csv"
echo "Long results:  ${OUTDIR}/clean_transfer_by_condition.csv"
echo "Separability:  ${OUTDIR}/clean_transfer_separability.json"
echo "Report:        ${OUTDIR}/report.json"
