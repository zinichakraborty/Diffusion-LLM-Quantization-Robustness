# SBATCH configurations
JOB_NAME_TEMPLATE = "Latency_evaluation_on_{model}"
NODES = 1
MEM = "500G"
GRES_GPU = "gpu:l40s:1"
TIME = "1:00:00"
LOG_FILE_PATH_TEMPLATE = "./logs/latency_evaluation_on_{model}.out"
MAIL_TYPE = "BEGIN,END,FAIL"

# Script template
SCRIPT_TEMPLATE = """
#!/bin/bash

module load uv

export HF_ALLOW_CODE_EVAL=1
export SLURM_INCLUDE_DIR=/opt/slurm/current/include
export SLURM_LIB_DIR=/opt/slurm/current/lib
eval "$(pixi shell-hook)"
srun pixi run python test_latency.py --model-dir "{model_dir}" --num-runs {num_runs} --warmup-runs {warmup_runs}
"""
