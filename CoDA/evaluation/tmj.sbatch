#!/bin/bash

#

#SBATCH --job-name=Task_evaluations

#SBATCH --nodes=1

#SBATCH --mem=500G

#SBATCH --gres=gpu:h100:1

#SBATCH --time=5:00:00

#SBATCH --output=./logs/%x_%j.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agupta965@gatech.edu            # e-mail address for notifications

 
########################

# User configuration

########################
 
# Model + paths (EDIT THESE)

MODEL="/storage/ice1/0/7/agupta965/research2/coda-3poch-gsm"

SAFE_MODEL_NAME="coda3poch_gsm"

RESULTS_DIR="./results"

LOG_DIR="./logs"
 
# Eval config

BATCH_SIZE=4
 
# Evaluation sets

EVALUATION_SETS=(

  "humaneval_instruct"

  "humaneval_plus"

  "mbpp_instruct"

  "mbpp_plus_instruct"

)
 
########################

# Environment setup

########################
 
module load uv
 
export HF_ALLOW_CODE_EVAL=1

export SLURM_INCLUDE_DIR=/opt/slurm/current/include

export SLURM_LIB_DIR=/opt/slurm/current/lib
 
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
 
########################

# Run evaluations

########################
 
for evaluation_set in "${EVALUATION_SETS[@]}"; do

  log_file="${LOG_DIR}/${evaluation_set}_evaluation_on_${SAFE_MODEL_NAME}.out"
 
  echo "Starting evaluation set: ${evaluation_set}"

  echo "Logging to: ${log_file}"
 
  args=(

    --model diffllm

    --model_args "pretrained=${MODEL},trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy"

    --tasks "${evaluation_set}"

    --device cuda

    --batch_size "${BATCH_SIZE}"

    --num_fewshot 0

    --output_path "${RESULTS_DIR}/${evaluation_set}_evaluation_on_${SAFE_MODEL_NAME}"

    --log_samples

    --confirm_run_unsafe_code

    --apply_chat_template true

  )
 
  # One-at-a-time; drop the "&" if you want strict sequential execution anyway.

  srun pixi run accelerate launch -m lm_eval "${args[@]}" &> "${log_file}"

done
 
echo "All evaluations finished."

 