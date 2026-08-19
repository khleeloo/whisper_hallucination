#!/bin/bash
#SBATCH --job-name=wh_failure_amp
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_failure_amp.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_failure_amp.err

set -euo pipefail

source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
STRESS_MAIN=${ROOT}/stress_eval_64pct
STRESS_UR=${ROOT}/stress_eval_64pct_ur_checkpoint-10000
CLEAN_BASE=${ROOT}/eval_validation/per_utterance_base_ckpt14000.csv
CLEAN_RR=${ROOT}/eval_64pct/per_utterance_rr_64pct_checkpoint-9375.csv
CLEAN_RU=${ROOT}/eval_64pct/per_utterance_ru_64pct_checkpoint-9375.csv
CLEAN_UR_GLOB='${ROOT}/eval_64pct/per_utterance_ur_64pct_checkpoint-10000*.csv'
CLEAN_UU_GLOB='${ROOT}/eval_64pct/per_utterance_uu_64pct_final*.csv'
OUT=${ROOT}/failure_amplification
mkdir -p "$OUT"

echo "=== Failure amplification analysis ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Python: $(which python)"
python --version

# Required clean files/globs. UR and UU may be stored as shards.
for f in "$CLEAN_BASE" "$CLEAN_RR" "$CLEAN_RU"; do
    [[ -s "$f" ]] || { echo "ERROR: missing clean CSV: $f" >&2; exit 2; }
done

UR_COUNT=$(compgen -G "${ROOT}/eval_64pct/per_utterance_ur_64pct_checkpoint-10000*.csv" | wc -l || true)
UU_COUNT=$(compgen -G "${ROOT}/eval_64pct/per_utterance_uu_64pct_final*.csv" | wc -l || true)
MAIN_STRESS_COUNT=$(find "$STRESS_MAIN" -maxdepth 1 -type f -name 'details_*.tsv' 2>/dev/null | wc -l || true)
UR_STRESS_COUNT=$(find "$STRESS_UR" -maxdepth 1 -type f -name 'details_*.tsv' 2>/dev/null | wc -l || true)

echo "Clean Base: $CLEAN_BASE"
echo "Clean RR:   $CLEAN_RR"
echo "Clean RU:   $CLEAN_RU"
echo "Clean UR files: $UR_COUNT"
echo "Clean UU files: $UU_COUNT"
echo "Main stress TSVs: $MAIN_STRESS_COUNT"
echo "UR checkpoint-10000 stress TSVs: $UR_STRESS_COUNT"

[[ "$UR_COUNT" -gt 0 ]] || { echo "ERROR: no clean UR checkpoint-10000 CSVs found" >&2; exit 2; }
[[ "$UU_COUNT" -gt 0 ]] || { echo "ERROR: no clean UU final CSVs found" >&2; exit 2; }
[[ "$MAIN_STRESS_COUNT" -gt 0 ]] || { echo "ERROR: no perturbed details TSVs in $STRESS_MAIN" >&2; exit 2; }

ARGS=(
    --perturbed_glob "$STRESS_MAIN/details_*.tsv"
    --clean "Base=$CLEAN_BASE"
    --clean "RR=$CLEAN_RR"
    --clean "RU=$CLEAN_RU"
    --clean "UR=${ROOT}/eval_64pct/per_utterance_ur_64pct_checkpoint-10000*.csv"
    --clean "UU=${ROOT}/eval_64pct/per_utterance_uu_64pct_final*.csv"
    --output_dir "$OUT"
    --bootstrap 10000
    --seed 42
)

# The manuscript's canonical UR stress run uses checkpoint-10000 in its own directory.
# Add it when present. The Python loader rejects duplicate condition/perturbation tags,
# preventing accidental mixing of two UR checkpoints.
if [[ "$UR_STRESS_COUNT" -gt 0 ]]; then
    ARGS+=(--perturbed_glob "$STRESS_UR/details_*.tsv")
fi

echo "=== Running analysis ==="
python -u analyze_failure_amplification.py "${ARGS[@]}"

echo ""
echo "=== Outputs ==="
ls -lh "$OUT"
echo "=== Complete ==="
