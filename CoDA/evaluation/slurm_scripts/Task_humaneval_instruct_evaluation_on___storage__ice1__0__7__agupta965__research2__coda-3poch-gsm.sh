#!/bin/bash
module load uv
export HF_ALLOW_CODE_EVAL=1
export SLURM_INCLUDE_DIR=/opt/slurm/current/include
export SLURM_LIB_DIR=/opt/slurm/current/lib
srun pixi run accelerate launch -m lm_eval \
  --model diffllm \
  --model_args "pretrained=/storage/ice1/0/7/agupta965/research2/coda-3poch-gsm,trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy" \
  --tasks humaneval_instruct \
  --device cuda \
  --batch_size 4 \
  --num_fewshot 0 \
  --output_path "results/humaneval_instruct_evaluation_on___storage__ice1__0__7__agupta965__research2__coda-3poch-gsm" \
  --log_samples \
  --confirm_run_unsafe_code \
  --apply_chat_template true