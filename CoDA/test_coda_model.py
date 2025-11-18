#!/usr/bin/env python3
import argparse
import os
import sys
import time
import math
import statistics
from typing import List, Optional

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="Salesforce/CoDA-v0-Instruct")
    p.add_argument("--torch-dtype", type=str, choices=["bfloat16","float16","float32"], default="float16")
    p.add_argument("--attn-impl", type=str, choices=["auto","eager","sdpa","flash_attention_2"], default="auto")
    p.add_argument("--num-runs", type=int, default=100)
    p.add_argument("--warmup-runs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=128)
    p.add_argument("--skip-generation", action="store_true")
    return p.parse_args()

args = parse_args()

# If user explicitly asks for float32, make sure math backend is allowed even before importing torch.
if args.torch_dtype == "float32":
    os.environ.setdefault("PYTORCH_SDP_BACKEND", "math")
    os.environ.setdefault("PYTORCH_SDP_DISABLE_BACKENDS", "flash,mem_efficient,cudnn")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# ---------------- dtype & backend helpers ----------------

def get_torch_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]

def _gpu_cc():
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()

def _force_safe_sdpa_on_pre_ampere():
    """
    Make sure a kernel exists on sm<80:
      - disable flash
      - disable mem_efficient (optional; set False to be strict)
      - enable math
    Apply both NEW and OLD APIs, and actually enter the new context manager globally.
    """
    maj, _ = _gpu_cc()
    if maj >= 8 or not torch.cuda.is_available():
        return

    # Try new API first (PyTorch 2.4+)
    try:
        from torch.nn.attention import sdpa_kernel
        # Enter context permanently (until process exit)
        _GLOBAL_SDPA_CTX = sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        _GLOBAL_SDPA_CTX.__enter__()
        print("[SDPA] Enabled math kernel (new API), disabled flash/mem_efficient for sm<80.")
    except Exception:
        pass

    # Also call legacy toggles to be safe
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDPA] Enabled math kernel (legacy API), disabled flash/mem_efficient for sm<80.")
    except Exception:
        pass

def _pick_dtype_for_device(requested_dtype: torch.dtype) -> torch.dtype:
    maj, _ = _gpu_cc()
    if maj < 8 and requested_dtype == torch.bfloat16:
        print("[Note] sm<80 detected; overriding bfloat16 -> float16 for better kernel coverage.")
        return torch.float16
    return requested_dtype

def _coerce_attn_impl(attn_impl: str) -> str:
    maj, _ = _gpu_cc()
    if maj < 8 and attn_impl in ("flash_attention_2", "auto"):
        print("[Note] sm<80 detected; coercing attn_implementation -> 'sdpa'.")
        return "sdpa"
    return attn_impl

# ---------------- manual attention (ultimate fallback) ----------------

def _to_additive_mask(mask: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype == torch.bool:
        add = torch.zeros_like(mask, dtype=target_dtype)
        add = add.masked_fill(~mask, float("-inf"))
        return add
    return mask.to(target_dtype)

def manual_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                dropout_p: float = 0.0, is_causal: bool = False,
                scale: Optional[float] = None) -> torch.Tensor:
    Dh = q.size(-1)
    s = (1.0 / math.sqrt(Dh)) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * s
    if is_causal:
        Lq, Lk = scores.size(-2), scores.size(-1)
        causal = torch.ones((Lq, Lk), device=scores.device, dtype=torch.bool).tril()
        while causal.dim() < scores.dim():
            causal = causal.unsqueeze(0)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        scores = scores + _to_additive_mask(attn_mask, scores.dtype)
    attn = torch.softmax(scores, dim=-1)
    # dropout only during training
    if dropout_p and q.requires_grad and q.training:
        attn = torch.dropout(attn, p=dropout_p, train=True)
    return torch.matmul(attn, v)

def patch_remote_attention_symbols():
    """
    Replace any remote module's reference to F.scaled_dot_product_attention
    with our manual_sdpa (works for fp16/fp32 on any GPU).
    """
    patched = []
    for name, mod in list(sys.modules.items()):
        try:
            modfile = getattr(mod, "__file__", "") or ""
        except Exception:
            continue
        if ("transformers_modules" in name or "transformers_modules" in modfile) and "attention" in name.lower():
            # direct symbol
            if hasattr(mod, "scaled_dot_product_attention"):
                try:
                    setattr(mod, "scaled_dot_product_attention", manual_sdpa)
                    patched.append(name)
                except Exception:
                    pass
            # alias case
            for k, v in list(getattr(mod, "__dict__", {}).items()):
                if callable(v) and getattr(v, "__name__", "") == "scaled_dot_product_attention":
                    try:
                        setattr(mod, k, manual_sdpa)
                        if name not in patched:
                            patched.append(name)
                    except Exception:
                        pass
    try:
        F.scaled_dot_product_attention = manual_sdpa  # last-resort net
    except Exception:
        pass
    if patched:
        print(f"[Patch] Overrode scaled_dot_product_attention in: {patched}")
    else:
        print("[Patch] No remote attention modules found to override (this is OK).")

# --------------- benchmarking ---------------

def measure_latency(model, num_runs: int = 100, warmup_runs: int = 10, seq_length: int = 128, batch_size: int = 1):
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)

    print(f"Running {warmup_runs} warmup iterations...")
    with torch.inference_mode():
        for _ in range(warmup_runs):
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Measuring latency over {num_runs} iterations...")
    latencies: List[float] = []
    with torch.inference_mode():
        for i in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                _ = model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_runs} iterations")
    return latencies

def test_generation(model, model_name: str):
    print("\n===== Testing Generation =====")
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (1, 32), device=device)
    if hasattr(model, "diffusion_generate"):
        print("===== generating with diffusion_generate =====")
        try:
            y = model.diffusion_generate(input_ids, num_steps=4, max_length=256)
            print("Generated sequence shape:", tuple(y.shape))
        except Exception as e:
            print("Error in diffusion_generate:", e)
    elif hasattr(model, "sample"):
        print("===== generating with model sample =====")
        try:
            y = model.sample(input_ids, steps=128, max_tokens=256)
            print("Generated sequence shape:", tuple(y.shape))
        except Exception as e:
            print("Error in sample:", e)
    else:
        print("===== Model methods:", [m for m in dir(model) if not m.startswith("_")])

# ---------------- main ----------------

def main():
    print(f"Loading model: {args.model_name}")
    req_dtype = get_torch_dtype(args.torch_dtype)

    # Device-aware settings
    _force_safe_sdpa_on_pre_ampere()
    torch_dtype = _pick_dtype_for_device(req_dtype)
    print(f"Using torch_dtype: {torch_dtype}")

    attn_impl = _coerce_attn_impl(args.attn_impl)

    # Ampere+ TF32 hint (no effect on sm75)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Prepare config with attn impl hint (some remote code reads from config)
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
        device_map="cuda",
        torch_dtype=torch_dtype,
        config=cfg if cfg is not None else None,
        **({} if cfg is not None else extra)
    )

    # Debug: device & quantization
    if torch.cuda.is_available():
        print("GPU CC:", _gpu_cc())
    try:
        qmods = [m for m in model.modules() if "Quant" in type(m).__name__]
        print(f"Quantized modules detected: {len(qmods)} | example: {type(qmods[0]).__name__ if qmods else 'N/A'}")
    except Exception:
        pass

    # Try a tiny probe call; if SDPA still errors, patch to manual attention and retry once.
    try:
        with torch.inference_mode():
            _ = model(input_ids=torch.randint(0, 1000, (1, 8), device="cuda"),
                      attention_mask=torch.ones(1, 8, device="cuda"))
    except RuntimeError as e:
        if "No available kernel" in str(e):
            print("[Fallback] SDPA reported 'No available kernel' — patching manual attention.")
            patch_remote_attention_symbols()
        else:
            raise

    model.eval()

    print("\n===== Latency Testing =====")
    L = measure_latency(model, args.num_runs, args.warmup_runs, args.seq_length, args.batch_size)

    print("\n===== Latency Results =====")
    print(f"  Mean: {statistics.mean(L):.2f} ms")
    print(f"  Std:  {statistics.stdev(L) if len(L) > 1 else 0.0:.2f} ms")
    print(f"  Min:  {min(L):.2f} ms")
    print(f"  Max:  {max(L):.2f} ms")
    print(f"  Num runs: {len(L)}")

    if not args.skip_generation:
        test_generation(model, args.model_name)

if __name__ == "__main__":
    main()
