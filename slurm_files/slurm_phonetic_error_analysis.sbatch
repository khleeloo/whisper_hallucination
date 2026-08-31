#!/bin/bash
#SBATCH --job-name=asr_phone_audit
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_phone_audit.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_phone_audit.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

INPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/training_error_risk_map/per_utterance_training_error_risk_map.csv"
OUTPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/training_error_risk_map/phonetic_analysis"

echo "=== Phonetic-error audit ==="
echo "Git commit: $(git rev-parse HEAD)"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "GPU allocation required by cluster policy; analysis itself is CPU-only."

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing cached risk-map output: ${INPUT}" >&2
  exit 2
fi

python - <<'PY'
try:
    import cmudict
    print("cmudict available")
except Exception as exc:
    raise SystemExit(
        "cmudict is required. In the llama environment run: pip install cmudict\n"
        f"Original import error: {exc}"
    )
PY

python -m py_compile phonetic_error_analysis.py
python phonetic_error_analysis.py \
    --input "${INPUT}" \
    --output_dir "${OUTPUT}" \
    --min_word_coverage 0.80 \
    --primary_phone_threshold 0.30

echo
echo "=== Primary summary ==="
cat "${OUTPUT}/summary_phonetic_analysis.csv"
echo
echo "=== Sensitivity ==="
cat "${OUTPUT}/sensitivity_phonetic_thresholds.csv"
echo
echo "=== Done ==="
