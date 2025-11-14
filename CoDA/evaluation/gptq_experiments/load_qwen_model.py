#!/usr/bin/env python3
# export_original_qwen.py
# Usage examples:
#   python export_original_qwen.py \
#     --model-id Qwen/Qwen2-1.5B-Instruct \
#     --dtype bf16 \
#     --out-root gptq_quantized_models
#
#   # If you prefer a different HF repo (e.g., Qwen2.5)
#   python export_original_qwen.py --model-id Qwen/Qwen2.5-1.5B-Instruct
#
# Output dir (example):
#   gptq_quantized_models/Qwen__Qwen2-1.5B-Instruct-bf16/

import argparse
import os
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from safetensors.torch import save_model

DTypeStr = Literal["bf16", "fp16", "fp32"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out-root", type=str, default="gptq_quantized_models")
    return p.parse_args()

def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")

def pick_torch_dtype(s: DTypeStr) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]

def sanitize_generation_config(model):
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return
    gc = model.generation_config
    try:
        if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
            gc.temperature = None
    except Exception:
        try:
            gc.temperature = 1.0
        except Exception:
            pass
    try:
        gc.validate()
    except Exception:
        try:
            model.generation_config = GenerationConfig()
        except Exception:
            pass

def export_original(model_id: str, dtype_str: DTypeStr, out_root: str) -> str:
    os.makedirs(out_root, exist_ok=True)
    torch_dtype = pick_torch_dtype(dtype_str)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True, revision="main")

    # Some Qwen variants don’t define pad_token; set it to eos to avoid warnings during batching.
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    # Keep export on CPU so no GPU is required here.
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cpu",
        revision="main",
    )

    sanitize_generation_config(model)

    out_dir = os.path.join(out_root, f"{safe_name(model_id)}-{dtype_str}")
    os.makedirs(out_dir, exist_ok=True)

    model.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    save_model(
        model,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    print(f"[OK] Exported original Qwen model to: {out_dir}")
    return out_dir

def main():
    args = parse_args()
    export_original(args.model_id, args.dtype, args.out_root)

if __name__ == "__main__":
    main()
