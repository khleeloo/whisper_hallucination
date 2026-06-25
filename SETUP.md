# Local Setup

This project is intended to run on the cluster with the `llama` conda
environment used by the SLURM scripts.

## Install

```bash
cd ~/code/whisper_hallucination
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama
pip install -r requirements.txt
```

`torchaudio.load` requires TorchCodec in the current `llama` environment. The
requirements file uses the CPU-only TorchCodec wheel because these scripts only
need CPU audio decoding and the default CUDA-enabled wheel may require CUDA
runtime libraries that are not present on login nodes.

If PyTorch is already installed with the correct CUDA build in `llama`, keep
that installation and install the remaining packages instead:

```bash
pip install transformers datasets accelerate peft evaluate jiwer sentence-transformers pandas numpy scikit-learn matplotlib seaborn tqdm safetensors
pip install torchcodec==0.14.0+cpu --index-url=https://download.pytorch.org/whl/cpu
```

## Common SLURM Entrypoints

Training:

```bash
sbatch slurm_files/slurm_16pct_train_sequential.sbatch
sbatch slurm_files/slurm_32pct_train_sequential.sbatch
sbatch slurm_files/slurm_64pct_train_sequential.sbatch
```

Evaluation:

```bash
sbatch slurm_files/slurm_eval_16pct.sbatch
sbatch slurm_files/slurm_eval_32pct.sbatch
sbatch slurm_files/slurm_eval_64pct.sbatch
```

The scripts assume project/data paths under:

```text
/scratch/vemotionsys/rmfrieske/whisper_hallucination
/scratch/vemotionsys/rmfrieske/datasets
```
