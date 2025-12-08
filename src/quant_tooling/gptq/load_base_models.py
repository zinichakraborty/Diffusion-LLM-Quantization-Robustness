import os
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from safetensors.torch import save_model

RESULTS_DIR = "gptq_quantized_models"

CODA_ID = "Salesforce/CoDA-v0-Instruct"
QWEN3_ID = "Qwen/Qwen3-1.7B"


def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")


def sanitize_generation_config(model):
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return
    gc = model.generation_config

    if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
        try:
            gc.temperature = None
        except Exception:
            gc.temperature = 1.0

    try:
        gc.validate()
    except Exception:
        model.generation_config = GenerationConfig()


def save_bf16_model(model_id: str, device: str = "cuda") -> str:
    print(f"Loading {model_id} in bf16 on {device}")

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
        attn_implementation="eager",
    )

    sanitize_generation_config(model)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_dir = os.path.join(RESULTS_DIR, f"{safe_name(model_id)}-bf16-base")
    os.makedirs(out_dir, exist_ok=True)

    model.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_model(
        model,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    del model
    del tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_dir

if __name__ == "__main__":
    print("Saving bf16 baseline: Salesforce CoDA")
    save_bf16_model(CODA_ID, device="cuda")

    print("Saving bf16 baseline: Qwen3-1.7B")
    save_bf16_model(QWEN3_ID, device="cuda")
