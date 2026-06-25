#!/bin/bash
#SBATCH --job-name=stress_new64
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_new64.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_stress_new64.err

set -euo pipefail

source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

BASE_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination
OUT_DIR=${BASE_DIR}/stress_eval_new_conditions_64pct
TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIP_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
MAX_SAMPLES=${MAX_SAMPLES:-1000}

mkdir -p "${OUT_DIR}"

CONFIG_NAMES=("base" "rr_64pct" "ru_64pct")
MODEL_DIRS=(
  "${BASE_DIR}/base/checkpoint-10000"
  "${BASE_DIR}/rr_64pct/checkpoint-9375"
  "${BASE_DIR}/ru_64pct/checkpoint-9375"
)

PERTURB_TYPES=(
  "silence"
  "leading_silence"
  "leading_silence"
  "speech_band_noise"
  "speech_band_noise"
)
PERTURB_AMPS=(
  "0.0"
  "0.0"
  "0.0"
  "0.5"
  "0.75"
)
PERTURB_DURS=(
  "0.0"
  "1.0"
  "3.0"
  "0.0"
  "0.0"
)

for idx in "${!CONFIG_NAMES[@]}"; do
  CONFIG="${CONFIG_NAMES[$idx]}"
  MODEL_DIR="${MODEL_DIRS[$idx]}"

  echo "============================================================"
  echo "Config: ${CONFIG}"
  echo "Model:  ${MODEL_DIR}"
  echo "============================================================"

  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Missing model directory: ${MODEL_DIR}" >&2
    exit 1
  fi

  for pidx in "${!PERTURB_TYPES[@]}"; do
    PTYPE="${PERTURB_TYPES[$pidx]}"
    PAMP="${PERTURB_AMPS[$pidx]}"
    PDUR="${PERTURB_DURS[$pidx]}"
    TAG="${PTYPE}_amp${PAMP}_dur${PDUR}"
    RESULT_JSON="${OUT_DIR}/results_${CONFIG}_${TAG}.json"
    DETAIL_TSV="${OUT_DIR}/details_${CONFIG}_${TAG}.tsv"

    if [[ -s "${RESULT_JSON}" && -s "${DETAIL_TSV}" ]]; then
      echo "Skipping existing ${CONFIG} ${TAG}"
      continue
    fi

    echo "Running ${CONFIG} ${TAG}"
    python evaluate_dual_metric.py \
      --model_dir "${MODEL_DIR}" \
      --test_tsv "${TEST_TSV}" \
      --clips_dir "${CLIP_DIR}" \
      --output_dir "${OUT_DIR}" \
      --config_name "${CONFIG}" \
      --perturb_type "${PTYPE}" \
      --perturb_amplitude "${PAMP}" \
      --perturb_duration "${PDUR}" \
      --max_samples "${MAX_SAMPLES}" \
      --batch_size 8 \
      --lm_models gpt2 \
      --num_log_examples 5
  done
done

echo "New-condition stress evaluation complete: ${OUT_DIR}"
