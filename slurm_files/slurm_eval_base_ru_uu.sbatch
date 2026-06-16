#!/bin/bash
#SBATCH --job-name=wh_eval
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_eval.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_eval.err
#SBATCH --mem 200G
#SBATCH --array=0-2  # 0=base, 1=ru, 2=uu

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1
cd /home/rmfrieske/whisper_hallucination

CONFIGS=("base" "ru" "uu")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
MODELS_BASE=/scratch/vemotionsys/rmfrieske/whisper_hallucination
# Use latest checkpoint for each config
case "${CONFIG}" in
    "base") CKPT=checkpoint-14000 ;;
    "ru") CKPT=checkpoint-10000 ;;
    "uu") CKPT=checkpoint-14000 ;;
    *) CKPT=final ;;
esac
MODEL_DIR=${MODELS_BASE}/${CONFIG}/${CKPT}
EVAL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results

echo "=== Evaluating config: ${CONFIG} (model: ${MODEL_DIR}) ==="

# Standard evaluation (no perturbation)
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name ${CONFIG} \
    --lm_models gpt2 Qwen/Qwen3-0.6B \
    --batch_size 32

echo "Evaluation complete for config: ${CONFIG}"
