import argparse
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from accelerate import init_empty_weights
from transformers import AutoModel, AutoConfig, AutoTokenizer
from optimum.gptq import load_quantized_model
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the model (base HF or GPTQ quantized). "
             "A model is treated as quantized if 'bit' is in the directory name.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2000,
        help="Number of forward passes to measure",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=200,
        help="Number of warmup runs before measuring",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for latency testing",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Sequence length for latency testing",
    )
    return parser.parse_args()

def _prepare_backend_for_gpu():
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=True,
            enable_math=True,
        )
    except Exception:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def _pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return torch.float16
    else:
        return torch.bfloat16

def load_gptq_model(args, torch_dtype, attn_impl):
    if args.gptq_model_dir is None:
        raise ValueError()

    gptq_dir = Path(args.gptq_model_dir)

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if attn_impl != "auto":
        cfg.attn_implementation = attn_impl

    base_model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        config=cfg,
        device_map=None,
    )

    model = load_quantized_model(
        base_model,
        save_folder=str(gptq_dir),
        device_map="auto",
    )
    return model


def load_base_model(args, torch_dtype, attn_impl):
    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if attn_impl != "auto":
            cfg.attn_implementation = attn_impl
    except Exception:
        cfg = None

    extra = {}
    if attn_impl != "auto":
        extra["attn_implementation"] = attn_impl
    elif torch_dtype == torch.float32:
        extra["attn_implementation"] = "sdpa"

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=args.device,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
        config=cfg if cfg is not None else None,
        **({} if cfg is not None else extra),
    )
    return model

def measure_latency(
    model,
    model_dir: Path,
    num_runs: int = 2000,
    warmup_runs: int = 200,
    batch_size: int = 1,
    seq_length: int = 1024,
):
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("No tokenizer found in model directory, using random inputs")

    if tokenizer is not None:
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
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device="cuda")
        attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                try:
                    _ = model(input_ids)
                except Exception:
                    raise

    torch.cuda.synchronize()

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
                except Exception:
                    raise

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)
    return latencies

def save_results(model_dir: Path, latencies: List[float]):
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
    model_dir = Path(args.model_dir).resolve()

    is_gptq = "bit" in model_dir.name

    _prepare_backend_for_gpu()
    torch_dtype = _pick_dtype()
    attn_impl = "sdpa"

    loader_args = SimpleNamespace(
        model_name=str(model_dir),
        gptq_model_dir=str(model_dir),
        device="cuda",
        attn_impl=attn_impl,
    )

    if is_gptq:
        model = load_gptq_model(loader_args, torch_dtype=torch_dtype, attn_impl=attn_impl)
    else:
        model = load_base_model(loader_args, torch_dtype=torch_dtype, attn_impl=attn_impl)

    model.eval()
    model.to("cuda")

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