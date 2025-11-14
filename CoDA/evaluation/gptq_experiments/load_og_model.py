#!/usr/bin/env python3
# export_original_coda.py
# Usage:
#   python export_original_coda.py \
#     --model-id Salesforce/CoDA-v0-Instruct \
#     --dtype bf16 \
#     --out-root gptq_quantized_models
#
# Output dir example:
#   gptq_quantized_models/Salesforce__CoDA-v0-Instruct-fp16

import argparse
import os
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from safetensors.torch import save_model


DTypeStr = Literal["bf16", "fp16", "fp32"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="Salesforce/CoDA-v0-Instruct")
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out-root", type=str, default="gptq_quantized_models")
    return p.parse_args()


def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")


def pick_torch_dtype(s: DTypeStr) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[s]


def sanitize_generation_config(model):
    """
    Avoid bad/invalid generation configs causing save errors.
    """
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return
    gc = model.generation_config

    # If sampling is off and temp==0, unset or normalize temperature
    try:
        if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
            gc.temperature = None  # preferred (won't serialize)
    except Exception:
        try:
            gc.temperature = 1.0
        except Exception:
            pass

    # Validate; if it still fails, just drop to a clean config
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

    # Keep load/save on CPU to avoid needing a GPU for export
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True, revision="main")

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cpu",
        revision="main",
    )

    # Make sure generation_config won't block saving
    sanitize_generation_config(model)

    # Directory name mirrors your quantized naming style
    tag = {"bf16": "fp16" if not torch.cuda.is_available() and torch_dtype == torch.bfloat16 else "bf16",
           "fp16": "fp16",
           "fp32": "fp32"}[dtype_str]
    # Note: the tag above keeps names predictable; feel free to simplify to dtype_str.

    out_dir = os.path.join(out_root, f"{safe_name(model_id)}-{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # Save config/tokenizer first
    model.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Save weights in a single safetensors file (HF-compatible)
    save_model(
        model,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},  # required for HF loaders
    )

    print(f"[OK] Exported original model to: {out_dir}")
    return out_dir


def main():
    args = parse_args()
    export_original(args.model_id, args.dtype, args.out_root)


if __name__ == "__main__":
    main()
