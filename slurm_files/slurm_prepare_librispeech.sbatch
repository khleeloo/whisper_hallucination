#!/bin/bash
#SBATCH --job-name=wh_libri_prep
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_libri_prep.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_libri_prep.err
#SBATCH --mem 32G

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

OUT=/scratch/vemotionsys/rmfrieske/datasets/librispeech_eval

python prepare_librispeech.py \
    --output_root ${OUT} \
    --splits test.clean test.other

status=$?
if [ $status -ne 0 ]; then
    echo "LibriSpeech prep FAILED (exit $status)"
    exit $status
fi
echo "LibriSpeech prep complete: ${OUT}"
