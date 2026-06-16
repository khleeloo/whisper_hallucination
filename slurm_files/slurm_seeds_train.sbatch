#!/bin/bash
#SBATCH --job-name=wh_seeds
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_seed.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_seed.err
#SBATCH --mem 300G
#SBATCH --array=0-9

# 5 key configs x 2 extra seeds (43, 44).
# Original seed 42 already covered by main + sweep runs.
#   0: base seed43       1: base seed44
#   2: uu_10 seed43      3: uu_10 seed44
#   4: rr_10 seed43      5: rr_10 seed44
#   6: uu_50 seed43      7: uu_50 seed44
#   8: rr_50 seed43      9: rr_50 seed44

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("base" "base" "uu_10" "uu_10" "rr_10" "rr_10" "uu_50" "uu_50" "rr_50" "rr_50")
SEEDS=(43       44       43       44       43       44       43       44       43       44)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
OUTPUT_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_seed${SEED}

echo "=== Seed run: config=${CONFIG} seed=${SEED} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
export PYTHONUNBUFFERED=1

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
    --seed ${SEED}

status=$?
if [ $status -ne 0 ]; then
    echo "Seed run FAILED: ${CONFIG} seed=${SEED} (exit $status)"
    exit $status
fi
echo "Seed run complete: ${CONFIG} seed=${SEED}"
