#!/bin/bash
#SBATCH --job-name=wh_failure_amp
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
STRESS_MAX_SAMPLES="${STRESS_MAX_SAMPLES:-1000}"

OUT=${ROOT}/failure_amplification
DUAL=${OUT}/dual_lm_sources
CLEAN_BASE=${OUT}/per_utterance_base_ckpt10000_1000.csv
CLEAN_RR=${ROOT}/eval_64pct/per_utterance_rr_64pct_checkpoint-9375.csv
CLEAN_RU=${ROOT}/eval_64pct/per_utterance_ru_64pct_checkpoint-9375.csv
mkdir -p "$OUT" "$DUAL"

echo "=== Structured-mismatch failure-signature analysis: Qwen3 + GPT2 ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python: $(which python)"
python --version

for f in "$TEST_TSV" "$CLEAN_BASE" "$CLEAN_RR" "$CLEAN_RU"; do
    [[ -s "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 2; }
done
[[ -d "$CLIPS_DIR" ]] || { echo "ERROR: missing clips directory: $CLIPS_DIR" >&2; exit 2; }

UR_COUNT=$(compgen -G "${ROOT}/eval_64pct/per_utterance_ur_64pct_checkpoint-10000*.csv" | wc -l || true)
UU_COUNT=$(compgen -G "${ROOT}/eval_64pct/per_utterance_uu_64pct_final*.csv" | wc -l || true)
MAIN_STRESS_COUNT=$(find "$STRESS_MAIN" -maxdepth 1 -type f -name 'details_*.tsv' 2>/dev/null | wc -l || true)
UR_STRESS_COUNT=$(find "$STRESS_UR" -maxdepth 1 -type f -name 'details_*.tsv' 2>/dev/null | wc -l || true)

echo "Test TSV: $TEST_TSV"
echo "Clips: $CLIPS_DIR"
echo "Stress max samples: $STRESS_MAX_SAMPLES"
echo "Clean Base: $CLEAN_BASE"
echo "Clean UR files: $UR_COUNT"
echo "Clean UU files: $UU_COUNT"
echo "Main stress TSVs: $MAIN_STRESS_COUNT"
echo "UR checkpoint-10000 stress TSVs: $UR_STRESS_COUNT"
echo "Primary plausibility LM: Qwen/Qwen3-0.6B"
echo "Parallel robustness LM: GPT2"
echo "Historical WAcc columns are ignored."

[[ "$UR_COUNT" -gt 0 ]] || { echo "ERROR: no clean UR checkpoint-10000 CSVs found" >&2; exit 2; }
[[ "$UU_COUNT" -gt 0 ]] || { echo "ERROR: no clean UU final CSVs found" >&2; exit 2; }
[[ "$MAIN_STRESS_COUNT" -gt 0 ]] || { echo "ERROR: no perturbed details TSVs in $STRESS_MAIN" >&2; exit 2; }

# Legacy stress TSVs stored only the first LM in a generic norm_plausibility
# column. They were generated with GPT2 first, so reuse that exact score and
# add explicit Qwen3-0.6B scores post hoc. Clean files with explicit per-LM
# columns are reused without rescoring those rows. This avoids re-running Whisper.
echo "=== Canonicalizing all sources to explicit Qwen3 + GPT2 columns ==="
RESCORE_ARGS=(
    --input_glob "$STRESS_MAIN/details_*.tsv"
    --input "$CLEAN_BASE"
    --input "$CLEAN_RR"
    --input "$CLEAN_RU"
    --input_glob "${ROOT}/eval_64pct/per_utterance_ur_64pct_checkpoint-10000*.csv"
    --input_glob "${ROOT}/eval_64pct/per_utterance_uu_64pct_final*.csv"
    --output_dir "$DUAL"
    --qwen_model Qwen/Qwen3-0.6B
    --gpt2_model gpt2
    --lm_batch_size "${LM_BATCH_SIZE:-8}"
    --legacy_first_lm gpt2
)
if [[ "$UR_STRESS_COUNT" -gt 0 ]]; then
    RESCORE_ARGS+=(--input_glob "$STRESS_UR/details_*.tsv")
fi
python -u rescore_dual_lm_outputs.py "${RESCORE_ARGS[@]}"

DUAL_STRESS_COUNT=$(find "$DUAL" -maxdepth 1 -type f -name 'details_*.tsv' | wc -l || true)
[[ "$DUAL_STRESS_COUNT" -gt 0 ]] || { echo "ERROR: no dual-LM stress TSVs produced" >&2; exit 2; }

echo "=== Running paired dual-LM analysis ==="
python -u analyze_failure_amplification_dual_lm.py \
    --perturbed_glob "$DUAL/details_*.tsv" \
    --clean "Base=$DUAL/per_utterance_base_ckpt10000_1000.csv" \
    --clean "RR=$DUAL/per_utterance_rr_64pct_checkpoint-9375.csv" \
    --clean "RU=$DUAL/per_utterance_ru_64pct_checkpoint-9375.csv" \
    --clean "UR=$DUAL/per_utterance_ur_64pct_checkpoint-10000*.csv" \
    --clean "UU=$DUAL/per_utterance_uu_64pct_final*.csv" \
    --test_tsv "$TEST_TSV" \
    --clips_dir "$CLIPS_DIR" \
    --stress_max_samples "$STRESS_MAX_SAMPLES" \
    --output_dir "$OUT" \
    --bootstrap 10000 \
    --seed 42

echo ""
echo "=== Outputs ==="
ls -lh "$OUT"
echo "=== Complete ==="
