#!/bin/bash
#SBATCH --job-name=quant_qwen
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

pixi run python create_qwen3_gptq_models.py
