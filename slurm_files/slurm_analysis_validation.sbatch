#!/bin/bash
#SBATCH --job-name=wh_val_analysis
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_analysis.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_analysis.err
#SBATCH --mem 64G

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

EVAL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation
PLOT_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/plots_validation
NOISY_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination

mkdir -p ${PLOT_DIR}

echo "=== Step 1: Merging per-utterance CSVs ==="
python -c "
import pandas as pd, os, glob
files = sorted(glob.glob('${EVAL_DIR}/per_utterance_*.csv'))
if not files:
    raise FileNotFoundError('No per_utterance_*.csv files found in ${EVAL_DIR}')
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df.to_csv('${EVAL_DIR}/per_utterance_metrics_whisper.csv', index=False)
print(f'Merged {len(files)} files into {len(df)} rows ({len(df.columns)} columns)')
print(f'Conditions: {sorted(df[\"noise_condition\"].unique())}')
"

echo "=== Step 2: Running aggregation analysis ==="
python analyze_validation.py \
    --per_utterance_dir ${EVAL_DIR} \
    --noisy_labels_dir ${NOISY_DIR} \
    --output_dir ${EVAL_DIR}

echo "=== Step 3: Generating plots ==="
python make_plots.py \
    --per_utterance_csv ${EVAL_DIR}/per_utterance_metrics_whisper.csv \
    --aggregate_csv ${EVAL_DIR}/aggregate_metrics_whisper.csv \
    --output_dir ${PLOT_DIR}

echo ""
echo "=== Pipeline complete! ==="
echo "Outputs:"
ls -lh ${EVAL_DIR}/
echo ""
ls -lh ${PLOT_DIR}/
