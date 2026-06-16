#!/bin/bash
#SBATCH --job-name=wh_train
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.err
#SBATCH --mem 300G
#SBATCH --array=0-4

# Array job: 0=base, 1=uu, 2=rr, 3=ru, 4=ur

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("base" "uu" "rr" "ru" "ur")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
OUTPUT_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}

echo "=== Training config: ${CONFIG} ==="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python finetune_whisper_lora.py \
    --noise_config ${CONFIG} \
    --data_dir ${DATA_DIR} \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_name openai/whisper-large-v3 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 500 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --seed 42

status=$?
if [ $status -ne 0 ]; then
    echo "Training FAILED for config: ${CONFIG} (exit $status)"
    exit $status
fi
echo "Training complete for config: ${CONFIG}"
