# Intelligent Diffusion LLM Quantization

[Intelligent_Quantization_for_Diffusion_LLMs.pdf](https://github.com/user-attachments/files/25829442/Intelligent_Quantization_for_Diffusion_LLMs.pdf)

We extensively use PACE ICE for our experiments. If this is your first time or you are new to PACE, please follow this [PACE ICE guide](https://github.com/guru-desh/Intro-To-PACE-ICE) made by Guru. These instructions assume that a user's PACE environment is configured according to the guide.

## Environment Setup

1. Run `salloc --gpus=1 --mem=100G --ntasks-per-node=24`
2. Install `pixi` via `curl -fsSL https://pixi.sh/install.sh | sh` (ONE TIME ONLY)
4. Run `module load cuda`
5. Run `pixi install` to setup virtual environment
6. Run `exit`

## GPTQ

To generate all models (Qwen and CoDA):

```sh
cd src/quant_tooling/gptq
sbatch load_models.sbatch
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
pixi run python run_benchmark_evals.py --model <HF model name or local path>  {optional: --results-dir <dir>}
pixi run python run_qwen3_benchmark_evals.py --models-dir <local path to all models> --results-dir <dir>

# same as prev but for latency
pixi run python run_latency_evals.py --models-dir <local path to gptq/HF saved model>
run python run_qwen3_latency_evals.py --models-dir <local path to all models>
```

Used to take in a model, saved locally in the HuggingFace format, and then dispatch SLURM jobs to either benchmark it on the 4 chosen coding benchmarks or run latency evaluation and return the mean latency (alongside the std dev) in a log file saved in `logs/`.
