#!/bin/bash
#SBATCH --job-name=wh100_train
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_100pct_train.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_100pct_train.err
#SBATCH --mem 256G

# Accept CONFIG as first argument: rr, ru, uu, or ur
CONFIG=${1:-rr}

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python:"
which python
python --version

echo "Torch:"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

echo "Torchrun:"
which torchrun || echo "torchrun not found"

echo "HF stack:"
python - <<'PY'
import huggingface_hub
import datasets
import transformers
import peft
import accelerate
import torch
print("hub:", huggingface_hub.__version__)
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("HF IMPORT TEST OK")
PY

cd /home/rmfrieske/whisper_hallucination

DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_100pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_100pct"

echo "=== Training ${CONFIG} with 100% noise on 4 GPUs ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/rmfrieske/whisper_hallucination/slurm_logs

# Preprocessing is handled internally by finetune_whisper_lora.py
# via _load_or_preprocess_fast() with disk caching.
# We just launch DDP training directly.

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
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --lora_r 32 \
    --lora_alpha 64 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --skip_audio_exists_check \
    --preprocessing_batch_size 4 \
    --seed 42

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete for ${CONFIG}_100pct"

    echo "=== Running eval for ${CONFIG}_100pct ==="
    EVAL_OUTPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_100pct"
    mkdir -p "${EVAL_OUTPUT}"

    python evaluate_whisper_validation.py \
        --model_dir "${OUTPUT_DIR}/final" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "${DATA_DIR}/test.tsv" \
        --clips_dir "${CLIPS_DIR}" \
        --output_dir "${EVAL_OUTPUT}" \
        --config_name "${CONFIG}_100pct" \
        --noise_condition "${CONFIG}" \
        --noise_ratio 1.0 \
        --batch_size 8 \
        --lm_models Qwen/Qwen3-0.6B
else
    echo "Training failed for ${CONFIG}_100pct with exit code ${EXIT_CODE}"
fi

echo "=== Done: ${CONFIG}_100pct (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
