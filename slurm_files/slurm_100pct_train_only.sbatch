#!/bin/bash
#SBATCH --job-name=wh100_train
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%x.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%x.err
#SBATCH --mem 256G

set -euo pipefail

CONFIG=${1:?Usage: sbatch slurm_100pct_train_only.sbatch <uu|ur|ru|rr>}
case "${CONFIG}" in
    uu|ur|ru|rr) ;;
    *) echo "ERROR: CONFIG must be one of uu, ur, ru, rr; got ${CONFIG}" >&2; exit 2 ;;
esac

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_100pct"
CACHE_DIR="${OUTPUT_DIR}/.dataset_cache"

cleanup_cache() {
    if [ "${KEEP_DATASET_CACHE:-0}" != "1" ]; then
        rm -rf "${CACHE_DIR}"
    fi
}
trap cleanup_cache EXIT

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/rmfrieske/whisper_hallucination/slurm_logs

echo "=== Training ${CONFIG}_100pct with 100% noise on 4 GPUs ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Cache dir: ${CACHE_DIR}"

echo "=== Removing stale preprocessing cache before launch ==="
rm -rf "${CACHE_DIR}"

echo "Python: $(which python)"
python --version
python - <<'PY'
import torch
import datasets
import transformers
import peft
import accelerate
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
PY

MASTER_PORT=$((20000 + RANDOM % 40000))
echo "=== Launching DDP training ==="
echo "MASTER_PORT=${MASTER_PORT}"

torchrun \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    finetune_whisper_lora.py \
    --data_dir "${DATA_DIR}" \
    --noise_config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --clips_dir "${CLIPS_DIR}" \
    --model_name openai/whisper-large-v3 \
    --num_epochs 5 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --final_wer_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --lora_r 32 \
    --lora_alpha 64 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --skip_audio_exists_check \
    --preprocessing_batch_size 4 \
    --seed 42

echo "=== Training complete for ${CONFIG}_100pct ==="
echo "=== Keeping model artifacts; dataset cache will be removed by EXIT trap ==="