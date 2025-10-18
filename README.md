# research2
Research repository for research2

## Environment Setup

1. Install `uv`: <https://docs.astral.sh/uv/getting-started/installation/>
2. Set the following environment variables:
```bash
export SLURM_INCLUDE_DIR=/opt/slurm/current/include
export SLURM_LIB_DIR=/opt/slurm/current/lib
```
3. Run `uv sync` to setup virtual environment

To test running `LLaDA-8B-Instruct`, run `uv run poe test_execute_model`

## Running Evaluations

1. TODO: Run `uv run python -m evaluation.run_evals`
