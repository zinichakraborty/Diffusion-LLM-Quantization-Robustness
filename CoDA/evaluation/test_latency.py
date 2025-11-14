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
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the GPTQ model")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of forward passes to measure")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs before measuring")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for latency testing")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length for latency testing")
    return parser.parse_args()

# --- pre-hook: fix star-unpacked tensors, handle [T,H] -> [1,T,H], dedupe kwargs ---
def _pre_hook(module, args, kwargs):
    # 0) Make sure we can mutate 'args'
    args = tuple(args)

    # A) Some Optimum paths star-unpack a 2D hidden_states [T,H] into T tensors of shape [H].
    # If we see "too many" positional args (way more than the forward signature), pack them back.
    if len(args) > 4 and all(isinstance(a, torch.Tensor) for a in args):
        try:
            hs = torch.stack(args, dim=0)        # -> [T, H]
            args = (hs,)                         # only hidden_states is positional
        except Exception:
            pass  # if stacking fails, fall through (won't fix this call)

    # B) Add a fake batch dim if Optimum passes packed [T,H]
    hs = args[0] if len(args) else None
    squeezed = isinstance(hs, torch.Tensor) and hs.dim() == 2
    if squeezed:
        hs = hs.unsqueeze(0)                     # -> [1, T, H]
        args = (hs,) + tuple(args[1:])
    module._coda_squeezed = squeezed

    # C) Dedupe kwargs that are already provided positionally
    if kwargs:
        kwargs = dict(kwargs)  # copy
        if len(args) >= 2 and "attention_mask" in kwargs:
            kwargs.pop("attention_mask")
        if len(args) >= 3 and "position_ids" in kwargs:
            kwargs.pop("position_ids")
        if len(args) >= 4 and "position_embeddings" in kwargs:
            kwargs.pop("position_embeddings")

    return args, (kwargs or {})

# --- post-hook: remove fake batch dim if we added it ---
def _post_hook(module, args, output):
    if getattr(module, "_coda_squeezed", False) and isinstance(output, torch.Tensor) and output.dim() == 3:
        return output.squeeze(0)                 # back to [T, H]
    return output

def attach_hooks(coda_model):
    """Attach hooks to handle CoDA model's special tensor shapes"""
    handles = []
    if hasattr(coda_model, 'model') and hasattr(coda_model.model, 'layers'):
        for block in coda_model.model.layers:
            handles.append(block.register_forward_pre_hook(_pre_hook, with_kwargs=True))
            handles.append(block.register_forward_hook(_post_hook))
    return handles

def _prepare_backend_for_gpu():
    # 1) safe SDPA fallback on pre-Ampere
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
    except Exception:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

def _pick_dtype():
    major, _ = torch.cuda.get_device_capability()
    return torch.float16 if major < 8 else "auto"

# in load_gptq_model(...)
def load_gptq_model(model_dir: Path):
    from transformers import AutoModel, AutoConfig
    _prepare_backend_for_gpu()

    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        setattr(cfg, "attention_kernel", "math")  # safe fallback for CoDA

    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=_pick_dtype(),   # fp16 on sm75
        config=cfg,
    )
    model.eval()
    return model

def measure_latency(model, model_dir: Path, num_runs: int = 100, warmup_runs: int = 10, batch_size: int = 32, seq_length: int = 512):
    """Measure forward pass latency"""
    # Load tokenizer from model directory
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        print("Loaded tokenizer from model directory")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Will use dummy input_ids")
    
    # Create input for testing
    
    # if tokenizer is not None:
    #     # Use tokenizer to create proper input
    #     text = "Hello, how are you? " * 20
    #     inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=seq_length, truncation=True)
    #     input_ids = inputs["input_ids"].to("cuda")
    #     attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
    # else:
        # Fallback: create dummy input_ids
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    with torch.inference_mode():
        for _ in range(warmup_runs):
            try:
                # CoDA models expect input_ids and attention_mask
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                # Try simpler forward pass
                try:
                    _ = model(input_ids)
                except Exception as e2:
                    print(f"Error during warmup: {e2}")
                    raise
    
    # Synchronize GPU
    torch.cuda.synchronize()
    
    # Actual latency measurements
    print(f"Measuring latency over {num_runs} iterations...")
    latencies: List[float] = []
    
    with torch.inference_mode():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            try:
                # CoDA models expect input_ids and attention_mask
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                # Try simpler forward pass
                try:
                    _ = model(input_ids)
                except Exception as e2:
                    print(f"Error during measurement: {e2}")
                    raise
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_runs} iterations")
    
    return latencies

def save_results(model_dir: Path, latencies: List[float]):
    """Save latency results to JSON file"""
    results = {
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "num_runs": len(latencies)
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
    
    # Load model
    model = load_gptq_model(model_dir)
    
    # Measure latency
    latencies = measure_latency(model, model_dir, args.num_runs, args.warmup_runs, args.batch_size, args.seq_length)
    
    # Save results
    save_results(model_dir, latencies)

if __name__ == "__main__":
    main()

