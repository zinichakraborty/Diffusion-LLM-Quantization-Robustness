# save_bf16_baselines.py
# pip install --upgrade "optimum>=1.21.0" transformers datasets safetensors

import os
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from safetensors.torch import save_model

RESULTS_DIR = "gptq_quantized_models_4"

CODA_ID = "Salesforce/CoDA-v0-Instruct"
QWEN3_ID = "Qwen/Qwen3-1.7B"  # HF id for Qwen3 1.7B dense model


def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")


def sanitize_generation_config(model):
    """Make sure generation_config is valid so save_pretrained doesn't explode."""
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return
    gc = model.generation_config

    # If sampling is off and temperature is 0, unset or reset it.
    if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
        try:
            gc.temperature = None
        except Exception:
            gc.temperature = 1.0

    try:
        gc.validate()
    except Exception:
        model.generation_config = GenerationConfig()  # clean fallback


def save_bf16_model(model_id: str, device: str = "cuda") -> str:
    """
    Load a model in bf16 and save it as a safetensors checkpoint under RESULTS_DIR,
    alongside GPTQ models.
    """
    print(f"Loading {model_id} in bf16 on {device}...")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    sanitize_generation_config(model)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(RESULTS_DIR, f"{safe_name(model_id)}-bf16-base")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving config & tokenizer to {out_dir}...")
    model.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print("Saving bf16 weights as model.safetensors...")
    save_model(
        model,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},  # so HF loaders recognize it
    )

    print(f"Done: {model_id} -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    print("======================================")
    print("Saving bf16 baseline: Salesforce CoDA")
    print("======================================")
    save_bf16_model(CODA_ID, device="cuda")

    print("======================================")
    print("Saving bf16 baseline: Qwen3-1.7B")
    print("======================================")
    save_bf16_model(QWEN3_ID, device="cuda")
