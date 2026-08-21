#!/bin/bash
#SBATCH --job-name=seamless_decision_util
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_seamless_decision_util.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_seamless_decision_util.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/vemotionsys/rmfrieske/huggingface_cache
mkdir -p "${HF_HOME}"

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
SOURCE=${ROOT}/seamless_m4t_v2_stress_pipeline_fixedwer/scored_outputs.csv
OUTDIR=${ROOT}/decision_utility_seamless

if [[ ! -f "${SOURCE}" ]]; then
  echo "Missing corrected Seamless baseline: ${SOURCE}" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

cat <<EOF
=== Decision utility: SeamlessM4T-v2 Large ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Baseline: ${SOURCE}
Output: ${OUTDIR}
Conditions: clean + full-noise 0.50 + full-noise 0.75
Collapse mitigation: DEV-learned dominant-output abstention
Collapse clean-DEV coverage floor: ${COLLAPSE_MIN_CLEAN_COVERAGE:-0.99}
Collapse lexicon cap: ${COLLAPSE_MAX_ENTRIES:-20}
Anti-repetition decoding: repetition_penalty=${REPETITION_PENALTY:-1.2}
Seed: ${SEED:-20260821}
EOF

python -u phenotype_targeted_mitigation.py \
  --model_type seamless \
  --baseline_source "${SOURCE}" \
  --output_dir "${OUTDIR}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --lm_batch_size "${LM_BATCH_SIZE:-8}" \
  --seed "${SEED:-20260821}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.2}" \
  --collapse_min_clean_coverage "${COLLAPSE_MIN_CLEAN_COVERAGE:-0.99}" \
  --collapse_max_entries "${COLLAPSE_MAX_ENTRIES:-20}" \
  --tgt_lang eng \
  "${EXTRA_ARGS[@]}"

echo
echo "=== Done ==="
echo "Collapse lexicon:  ${OUTDIR}/collapse_lexicon_dev.csv"
echo "Collapse TEST:     ${OUTDIR}/collapse_rejector_test_summary.csv"
echo "Anti-rep TEST:     ${OUTDIR}/antirep_test_summary.csv"
echo "Anti-rep deltas:   ${OUTDIR}/antirep_vs_baseline.csv"
