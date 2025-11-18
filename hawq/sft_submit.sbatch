#!/bin/bash

#SBATCH --job-name=Task
#SBATCH --nodes=1 --ntasks-per-node=24 --gres=gpu:A40:2
#SBATCH --time=8:00:00
#SBATCH --output=./logs/task.out
#SBATCH --mail-type=NONE

cd $SLURM_SUBMIT_DIR
cd sft
pixi run python hawq_sft.py --track_hessian