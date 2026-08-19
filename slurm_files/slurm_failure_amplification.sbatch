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

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

STRESS_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_eval_64pct
CLEAN_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct
OUT=/scratch/vemotionsys/rmfrieske/whisper_hallucination/failure_amplification

mkdir -p "$OUT"

echo "=== Failure amplification analysis ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Python: $(which python)"
python --version
echo "Stress dir: $STRESS_DIR"
echo "Clean dir:  $CLEAN_DIR"
echo "Output dir: $OUT"

echo "=== Input check ==="
if [ ! -d "$STRESS_DIR" ]; then
    echo "ERROR: missing stress directory: $STRESS_DIR" >&2
    exit 2
fi
if [ ! -d "$CLEAN_DIR" ]; then
    echo "ERROR: missing clean directory: $CLEAN_DIR" >&2
    exit 2
fi

STRESS_COUNT=$(find "$STRESS_DIR" -maxdepth 1 -type f -name 'details_*.tsv' | wc -l)
CLEAN_COUNT=$(find "$CLEAN_DIR" -maxdepth 1 -type f -name 'details_*.tsv' | wc -l)
echo "Stress detail TSVs: $STRESS_COUNT"
echo "Clean detail TSVs:  $CLEAN_COUNT"

if [ "$STRESS_COUNT" -eq 0 ]; then
    echo "ERROR: no details_*.tsv files found in $STRESS_DIR" >&2
    exit 2
fi
if [ "$CLEAN_COUNT" -eq 0 ]; then
    echo "ERROR: no details_*.tsv files found in $CLEAN_DIR" >&2
    exit 2
fi

echo "=== Running analysis ==="
python -u analyze_failure_amplification.py \
    --details_glob "$STRESS_DIR/details_*.tsv" \
    --details_glob "$CLEAN_DIR/details_*.tsv" \
    --output_dir "$OUT" \
    --bootstrap 10000 \
    --seed 42

echo ""
echo "=== Outputs ==="
ls -lh "$OUT"

echo "=== Complete ==="
