#!/bin/bash
#SBATCH --job-name=asr_risk_map
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_risk_map.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_risk_map.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/training_error_risk_map"

echo "=== Training-error structure risk-map experiment ==="
echo "Primary comparison: RR vs RU vs UR vs UU, all at 64% corruption"
echo "Git commit: $(git rev-parse HEAD)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
python - <<'PY'
import torch
import pandas
import sklearn
import transformers
import peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
PY

python -m py_compile training_error_risk_map.py

# First decisive test: keep corruption prevalence fixed and vary only its structure.
# We deliberately do not include the 16/32% survivor checkpoints here because their
# training-step comparability needs to be verified before claiming a dose response.
python training_error_risk_map.py \
    --output_dir "${OUTPUT_DIR}" \
    --conditions rr_64 ru_64 ur_64 uu_64 \
    --max_samples 300 \
    --batch_size 8 \
    --qwen_batch_size 8 \
    --seed 20260828 \
    --perturbations none full_noise_amp0.5_dur0.0

echo "=== 64% structure comparison ==="
cat "${OUTPUT_DIR}/structure_64pct.csv"
echo
echo "=== Done ==="
