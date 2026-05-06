#!/bin/bash
#SBATCH --job-name=wh_sweep_eval
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_sweep_eval.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_sweep_eval.err
#SBATCH --mem 200G
#SBATCH --array=0-12

# Sweep eval: 0=base, 1-6=UU sweep, 7-12=RR sweep
# Maps to: base, uu_01, uu_02, uu_05, uu_10, uu_20, uu_50,
#           rr_01, rr_02, rr_05, rr_10, rr_20, rr_50

CONFIGS=("base" \
         "uu_01" "uu_02" "uu_05" "uu_10" "uu_20" "uu_50" \
         "rr_01" "rr_02" "rr_05" "rr_10" "rr_20" "rr_50")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Model dir: base uses /base/final, sweep uses /sweep_XX/final
if [ "$CONFIG" == "base" ]; then
    MODEL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/base/final
else
    MODEL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/sweep_${CONFIG}/final
fi

module load Anaconda3
conda init
source activate llama

cd /home/rmfrieske/whisper_hallucination

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
EVAL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results

echo "=== Sweep eval: ${CONFIG} ==="
echo "Model: ${MODEL_DIR}"

# Standard evaluation only (dose-response doesn't need perturbation)
python evaluate_dual_metric.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${DATA_DIR}/test.tsv \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${EVAL_DIR} \
    --config_name sweep_${CONFIG} \
    --lm_model Qwen/Qwen3-1.7B \
    --batch_size 8

echo "Sweep eval complete for: ${CONFIG}"
