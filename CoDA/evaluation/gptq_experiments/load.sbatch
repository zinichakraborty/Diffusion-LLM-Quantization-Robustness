#!/bin/bash
#SBATCH --job-name=load_qwen
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

export HF_HOME="${HF_HOME:-/storage/ice1/6/6/$USER/caches/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$TRANSFORMERS_CACHE"

pixi run python load_og_models.py
