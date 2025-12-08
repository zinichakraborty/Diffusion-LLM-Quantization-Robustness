# Eval configurations
BATCH_SIZE = 8

# SBATCH configurations
JOB_NAME_TEMPLATE = "Task_{evaluation_set}_evaluation_on_{model}"
NODES = 1
MEM = "500G"
GRES_GPU = "gpu:h100:1"
TIME = "5:00:00"
LOG_FILE_PATH_TEMPLATE = "./logs/{evaluation_set}_evaluation_on_{model}.out"
MAIL_TYPE = "BEGIN,END,FAIL"

# Script template for each evaluation set
SCRIPT_TEMPLATE = """#!/bin/bash
module load uv

export HF_ALLOW_CODE_EVAL=1
export SLURM_INCLUDE_DIR=/opt/slurm/current/include
export SLURM_LIB_DIR=/opt/slurm/current/lib

srun pixi run accelerate launch -m lm_eval \\
  --model diffllm \\
  --model_args "pretrained={model},trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy" \\
  --tasks {evaluation_set} \\
  --device cuda \\
  --batch_size {batch_size} \\
  --num_fewshot 0 \\
  --output_path "{results_dir}/{evaluation_set}_evaluation_on_{safe_model_name}" \\
  --log_samples \\
  --confirm_run_unsafe_code \\
  --apply_chat_template true
"""

EVALUATION_SETS = ["humaneval_instruct", "humaneval_plus", "mbpp_instruct", "mbpp_plus_instruct"]
