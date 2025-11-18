# research2
Research repository for research2.

## Environment Setup

1. Run `salloc --gpus=1 --mem=100G --ntasks-per-node=24`
2. Install `pixi` via `curl -fsSL https://pixi.sh/install.sh | sh` (ONE TIME ONLY)
4. Run `module load cuda`
5. Run `pixi install` to setup virtual environment
6. Run `exit`

To test running `LLaDA-8B-Instruct`, run `pixi run poe test_execute_model`

## Running Evaluations

1. TODO: Run `pixi run python -m evaluation.run_evals`
