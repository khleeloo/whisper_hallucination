#!/bin/bash
#SBATCH --job-name=asr_shallow_pf
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_shallow_pf.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_shallow_pf.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

INPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/training_error_risk_map/per_utterance_training_error_risk_map.csv"
OUTPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/training_error_risk_map/shallow_pf"

echo "=== Official SHALLOW PF audit ==="
echo "Git commit: $(git rev-parse HEAD)"
echo "SHALLOW source commit: 47de7b6646a1a18735bf0860efa84f70d2e2ef06"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "GPU allocation required by cluster policy; SHALLOW PF itself is CPU-only."

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing cached risk-map output: ${INPUT}" >&2
  exit 2
fi

python - <<'PY'
try:
    import jellyfish
    print("jellyfish already available", getattr(jellyfish, "__version__", "version-unreported"))
except Exception:
    raise SystemExit(3)
PY
STATUS=$?
if [[ ${STATUS} -eq 3 ]]; then
  echo "Installing official SHALLOW dependency jellyfish==1.2.0"
  python -m pip install --quiet jellyfish==1.2.0
fi

python - <<'PY'
import jellyfish
assert hasattr(jellyfish, "metaphone")
assert hasattr(jellyfish, "hamming_distance")
assert hasattr(jellyfish, "levenshtein_distance")
assert hasattr(jellyfish, "jaro_winkler_similarity")
print("SHALLOW PF jellyfish API preflight OK")
PY

python -m py_compile shallow_pf_analysis.py
python shallow_pf_analysis.py \
    --input "${INPUT}" \
    --output_dir "${OUTPUT}" \
    --bootstrap_reps 10000 \
    --seed 20260831

echo
echo "=== SHALLOW PF summary CSV ==="
cat "${OUTPUT}/summary_shallow_pf.csv"
echo
echo "=== Pairwise contrasts ==="
cat "${OUTPUT}/pairwise_shallow_pf_contrasts.csv"
echo
echo "=== Done ==="
