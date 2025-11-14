#!/bin/bash
#SBATCH --job-name=export-unquant
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=logs/export-%j.out
#SBATCH --error=logs/export-%j.err

set -euo pipefail
mkdir -p logs

# (Optional) use node-local/scratch caches to avoid $HOME quota
export HF_HOME="${HF_HOME:-/storage/ice1/6/6/$USER/caches/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$TRANSFORMERS_CACHE"

# (Optional) many clusters export CUDA_VISIBLE_DEVICES automatically when using --gres
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run inside the allocation (no extra srun needed)
pixi run python load_qwen_model.py
