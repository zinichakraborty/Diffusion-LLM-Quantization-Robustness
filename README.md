# research2
Research repository for research2

## Environment Setup

1. Run `salloc --gpus=1 --mem=100G --ntasks-per-node=24`
2. Install `uv`: <https://docs.astral.sh/uv/getting-started/installation/>. On PACE, do `module load uv` 
3. Set the following environment variables:
```bash
export SLURM_INCLUDE_DIR=/opt/slurm/current/include
export SLURM_LIB_DIR=/opt/slurm/current/lib
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
export CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=70;75;80;86;89;90 -DUSER_CUDA_ARCH_LIST=7.0\;7.5\;8.0\;8.6\;8.9\;9.0+PTX'
```
4. Run `module load cuda`
5. Run `uv sync --no-build-isolation` to setup virtual environment
6. Run `exit`

To test running `LLaDA-8B-Instruct`, run `uv run poe test_execute_model`

## Running Evaluations

1. TODO: Run `uv run python -m evaluation.run_evals`
