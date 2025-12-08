#!/bin/bash

#SBATCH --job-name=quant_qwen_and_coda
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

pixi run python load_base_models.py
pixi run python create_coda_gptq_models.py
pixi run python create_qwen3_gptq_models.py