#!/bin/bash
#SBATCH --job-name=qwen3-gptq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1         # FIXED: correct gres syntax
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_gptq_%j.out
#SBATCH --error=logs/qwen3_gptq_%j.err

echo "SLURM job started at: $(date)"

# ---------------- ENV ----------------
mkdir -p logs

# HuggingFace caches — good
export HF_HOME="/storage/ice1/6/6/$USER/caches/hf"
mkdir -p "$HF_HOME/hub"
unset TRANSFORMERS_CACHE   

# Pixi environment — correct
eval "$(pixi shell-hook)"

# ---------------- PARAMS ----------------
MODEL_ID="Qwen/Qwen3-1.7B"
BITS="2 3 4 8"
DATASET="wikitext2"
RESULTS_DIR="gptq_quantized_models_3"

# ---------------- RUN ----------------
pixi run python create_qwen3_models.py \
    --model-id "${MODEL_ID}" \
    --bits ${BITS} \
    --calib-dataset "${DATASET}" \
    --results-dir "${RESULTS_DIR}" \
    --device "cuda"

echo "SLURM job finished at: $(date)"
