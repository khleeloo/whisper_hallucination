#!/bin/bash
# Submit evaluation for all 4 training runs
# Handles corrupted checkpoints by reconstructing missing adapter_config.json

ACCT="#SBATCH --account=vemotionsys"
CONDA_INIT="source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh && conda activate llama"

# Reconstruct adapter_config.json for 24pct checkpoints (weights exist, config missing)
for config in ru_24pct rr_24pct; do
    CKPT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${config}/checkpoint-2000"
    if [ -f "${CKPT}/adapter_model.safetensors" ] && [ ! -f "${CKPT}/adapter_config.json" ]; then
        echo "Reconstructing adapter_config.json for ${config}..."
        cp /scratch/vemotionsys/rmfrieske/whisper_hallucination/ru_16pct/checkpoint-1000/adapter_config.json "${CKPT}/adapter_config.json"
    fi
done

EVAL_PY="/home/rmfrieske/whisper_hallucination/evaluate_whisper_validation.py"
BASE="openai/whisper-large-v3"
TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTDIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation"
BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"

sbatch <<SBATCH
#!/bin/bash
${ACCT}
#SBATCH --job-name=eval_ru_16pct
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_ru_16pct.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_ru_16pct.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
${CONDA_INIT}
python ${EVAL_PY} \
    --model_dir ${BASE_DIR}/ru_16pct/checkpoint-2000 \
    --base_model ${BASE} \
    --test_tsv ${TSV} \
    --clips_dir ${CLIPS} \
    --output_dir ${OUTDIR} \
    --config_name ru_16pct \
    --noise_condition ru \
    --noise_ratio 0.16 \
    --batch_size 8
SBATCH

sbatch <<SBATCH
#!/bin/bash
${ACCT}
#SBATCH --job-name=eval_rr_16pct
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_rr_16pct.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_rr_16pct.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
${CONDA_INIT}
python ${EVAL_PY} \
    --model_dir ${BASE_DIR}/rr_16pct/checkpoint-2000 \
    --base_model ${BASE} \
    --test_tsv ${TSV} \
    --clips_dir ${CLIPS} \
    --output_dir ${OUTDIR} \
    --config_name rr_16pct \
    --noise_condition rr \
    --noise_ratio 0.16 \
    --batch_size 8
SBATCH

sbatch <<SBATCH
#!/bin/bash
${ACCT}
#SBATCH --job-name=eval_ru_24pct
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_ru_24pct.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_ru_24pct.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
${CONDA_INIT}
python ${EVAL_PY} \
    --model_dir ${BASE_DIR}/ru_24pct/checkpoint-2000 \
    --base_model ${BASE} \
    --test_tsv ${TSV} \
    --clips_dir ${CLIPS} \
    --output_dir ${OUTDIR} \
    --config_name ru_24pct \
    --noise_condition ru \
    --noise_ratio 0.24 \
    --batch_size 8
SBATCH

sbatch <<SBATCH
#!/bin/bash
${ACCT}
#SBATCH --job-name=eval_rr_24pct
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_rr_24pct.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_rr_24pct.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
${CONDA_INIT}
python ${EVAL_PY} \
    --model_dir ${BASE_DIR}/rr_24pct/checkpoint-2000 \
    --base_model ${BASE} \
    --test_tsv ${TSV} \
    --clips_dir ${CLIPS} \
    --output_dir ${OUTDIR} \
    --config_name rr_24pct \
    --noise_condition rr \
    --noise_ratio 0.24 \
    --batch_size 8
SBATCH

echo "All 4 eval jobs submitted."
