import argparse
import json
import time
import torch
from pathlib import Path
from typing import List
import statistics
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the GPTQ model (config + model.safetensors)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of forward passes to measure")
    parser.add_argument("--warmup-runs", type=int, default=10,
                        help="Number of warmup runs before measuring")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for latency testing")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length for latency testing")
    return parser.parse_args()


# ----------------- GPU backend helpers (generic, keep these) ----------------- #

def _prepare_backend_for_gpu():
    # Try to configure SDPA kernels safely
    try:
        torch.backends.cuda.sdp_kernel(
            enable_flash=False,      # flash may not exist on older GPUs
            enable_mem_efficient=True,
            enable_math=True,
        )
    except Exception:
        # Older PyTorch APIs
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def _pick_dtype():
    # Qwen3 is bf16-native on modern GPUs, but fp16 is safer for pre-Ampere.
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return torch.float16      # sm70/sm75, etc.
    else:
        return torch.bfloat16     # Ampere+ works well with bf16


# ----------------- Model loading (Qwen3 / generic HF) ----------------- #

def load_gptq_model(model_dir: Path):
    from transformers import AutoModel, AutoConfig

    _prepare_backend_for_gpu()

    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)

    # NOTE: For CoDA we forced cfg.attention_kernel = "math" here.
    # For Qwen3 we do NOT override attention_kernel; use model's own config.

    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=_pick_dtype(),
        config=cfg,
    )
    model.eval()
    return model


# ----------------- Latency measurement ----------------- #

def measure_latency(
    model,
    model_dir: Path,
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 32,
    seq_length: int = 512,
):
    """Measure forward pass latency for a GPTQ-quantized Qwen3 model."""
    from transformers import AutoTokenizer

    # Try to use the tokenizer from the model dir (this matches how you saved Qwen3)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Loaded tokenizer from model directory")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Falling back to dummy input_ids")

    if tokenizer is not None:
        # Use real tokenization to get realistic shapes
        text = "Hello, how are you? " * 20
        inputs = tokenizer(
            [text] * batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_length,
        )
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
    else:
        # Fallback: random tokens; fine for latency benchmarking
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device="cuda")
        attention_mask = torch.ones_like(input_ids)

    # Warmup
    print(f"Running {warmup_runs} warmup iterations...")
    with torch.inference_mode():
        for _ in range(warmup_runs):
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                try:
                    _ = model(input_ids)
                except Exception as e2:
                    print(f"Error during warmup: {e2}")
                    raise

    torch.cuda.synchronize()

    # Measure
    print(f"Measuring latency over {num_runs} iterations...")
    latencies: List[float] = []

    with torch.inference_mode():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                try:
                    _ = model(input_ids)
                except Exception as e2:
                    print(f"Error during measurement: {e2}")
                    raise

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_runs} iterations")

    return latencies


# ----------------- Saving results ----------------- #

def save_results(model_dir: Path, latencies: List[float]):
    """Save latency results to a timestamped JSON file in model_dir."""
    results = {
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "num_runs": len(latencies),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = model_dir / f"latency_result_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nLatency Results:")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  Min:  {results['min_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")
    print(f"\nResults saved to {output_path}")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    print(f"Loading GPTQ model from: {model_dir}")
    model = load_gptq_model(model_dir)

    latencies = measure_latency(
        model,
        model_dir,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )

    save_results(model_dir, latencies)


if __name__ == "__main__":
    main()
