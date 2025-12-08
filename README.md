# research2
Research repository for research2.

## Environment Setup

1. Run `salloc --gpus=1 --mem=100G --ntasks-per-node=24`
2. Install `pixi` via `curl -fsSL https://pixi.sh/install.sh | sh` (ONE TIME ONLY)
4. Run `module load cuda`
5. Run `pixi install` to setup virtual environment
6. Run `exit`

To test running `LLaDA-8B-Instruct`, run `pixi run poe test_execute_model`

## GPTQ

@Guru fill in
SBATCH script to load base model and Qwen3 quantized models:
```
#!/bin/bash
#SBATCH --job-name=quant_qwen
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

pixi run python load_base_models.py
pixi run python create_qwen3_gptq_models.py
```

## HAWQ

To run sensitivity calculation (Algorithm 1, 2):

```sh
cd src/quant_tooling/hawq/sft
pixi run python sens_gen.py --model_name Salesforce/CoDA-v0-Instruct --task wikitext2
```

It will run 1 pass over the specified dataset (`--task` attribute) and generate sensitivities for each module/block layer in the specified model. It is intended for use with `Salesforce CoDA` so you can either specify the HuggingFace link, or the path to a local directory created by HuggingFace's `{model/tokenizer}.save_pretrained(dir)` if a pretrained model is desired to be used. 

The dictionary of {layer: sensitivity} will be printed to console. 

Algorithm 3:

```sh
cd src/quant_tooling/hawq/quantlib
pixi run python main.py --sensitivities <path_to.json> --splits <% 16>/<% 8>/<% 4> --save-dir <dir> --modelname <huggingface name or local path>
```

Used to ingest a HuggingFace (or locally saved HF) model, quantize it to mixed precision in accordance with the splits argument, and then save it to the specified directory. Implements Algorithm 3 from our writeup. 

## Evaluation Scripts

```sh
cd src/evaluation

# run benchmark on HumanEval, MBPP benchmarks by dispatching SLURM jobs
pixi run python run_benchmark_evals.py --model <HF model name/local path>  {optional: --results-dir <dir>}

# same as prev but for latency
pixi run python run_latency_evals.py --models-dir <local path to gptq/HF saved model>
```

Used to take in a model, saved locally in the HuggingFace format, and then dispatch SLURM jobs to either benchmark it on the 4 chosen coding benchmarks or run latency evaluation and return the mean latency (alongside the std dev). 
