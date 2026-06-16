#!/bin/bash
#SBATCH --job-name=wh_sweep
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_sweep.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_sweep.err
#SBATCH --mem 300G
#SBATCH --array=0-11

# Sweep: 2 noise types x 6 ratios = 12 configs
# Index mapping:
#  0=uu_01  1=uu_02  2=uu_05  3=uu_10  4=uu_20  5=uu_50
#  6=rr_01  7=rr_02  8=rr_05  9=rr_10 10=rr_20 11=rr_50

CONFIGS=("uu_01" "uu_02" "uu_05" "uu_10" "uu_20" "uu_50" \
         "rr_01" "rr_02" "rr_05" "rr_10" "rr_20" "rr_50")

# Extract noise type from config name (part before _)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NOISE_TYPE=${CONFIG%%_*}

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
OUTPUT_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/sweep_${CONFIG}

echo "=== Sweep training: ${CONFIG} (type=${NOISE_TYPE}) ==="
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
    echo "Sweep training FAILED for: ${CONFIG} (exit $status)"
    exit $status
fi
echo "Sweep training complete for: ${CONFIG}"
