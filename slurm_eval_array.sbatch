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
#SBATCH --array=0-4

# Array job: 0=base, 1=uu, 2=rr, 3=ru, 4=ur

module load Anaconda3
conda init
source activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("base" "uu" "rr" "ru" "ur")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
MODEL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}/final
EVAL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results

echo "=== Evaluating config: ${CONFIG} ==="

# Standard evaluation (no perturbation)
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name ${CONFIG} \
    --lm_model Qwen/Qwen3-1.7B \
    --batch_size 8

# Perturbation: onset noise 0.1 amplitude, 0.5s
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name ${CONFIG} \
    --lm_model Qwen/Qwen3-1.7B \
    --batch_size 8 \
    --perturb_type onset_noise \
    --perturb_amplitude 0.1 \
    --perturb_duration 0.5

# Perturbation: onset noise 0.5 amplitude, 0.5s
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name ${CONFIG} \
    --lm_model Qwen/Qwen3-1.7B \
    --batch_size 8 \
    --perturb_type onset_noise \
    --perturb_amplitude 0.5 \
    --perturb_duration 0.5

# Perturbation: full noise 0.1 amplitude
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name ${CONFIG} \
    --lm_model Qwen/Qwen3-1.7B \
    --batch_size 8 \
    --perturb_type full_noise \
    --perturb_amplitude 0.1

# Perturbation: full noise 0.5 amplitude
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name ${CONFIG} \
    --lm_model Qwen/Qwen3-1.7B \
    --batch_size 8 \
    --perturb_type full_noise \
    --perturb_amplitude 0.5

echo "Evaluation complete for config: ${CONFIG}"
