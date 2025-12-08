import os
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from optimum.gptq import GPTQQuantizer
from safetensors.torch import save_model

BITS = [2, 3, 4, 8]

QWEN3_ID = "Qwen/Qwen3-1.7B"

CALIBRATION_DATASET = "wikitext2"
RESULTS_DIR = "gptq_quantized_models"


def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")

def _qwen3_pre_hook(module, args, kwargs):
    args = tuple(args)

    if len(args) > 4 and all(isinstance(a, torch.Tensor) for a in args):
        try:
            hs = torch.stack(args, dim=0)
            args = (hs,)
        except Exception:
            pass

    hs = args[0] if len(args) else None
    squeezed = isinstance(hs, torch.Tensor) and hs.dim() == 2
    if squeezed:
        hs = hs.unsqueeze(0)
        args = (hs,) + tuple(args[1:])
    module._qwen3_squeezed = squeezed

    if kwargs:
        kwargs = dict(kwargs)
        if len(args) >= 2 and "attention_mask" in kwargs:
            kwargs.pop("attention_mask")
        if len(args) >= 3 and "position_ids" in kwargs:
            kwargs.pop("position_ids")
        if len(args) >= 4 and "position_embeddings" in kwargs:
            kwargs.pop("position_embeddings")

    return args, (kwargs or {})

def _qwen3_post_hook(module, args, output):
    if getattr(module, "_qwen3_squeezed", False) and isinstance(output, torch.Tensor) and output.dim() == 3:
        return output.squeeze(0)
    return output


def attach_qwen3_hooks(m):
    handles = []

    if hasattr(m, "model") and hasattr(m.model, "layers"):
        blocks = m.model.layers
    elif hasattr(m, "layers"):
        blocks = m.layers
    else:
        raise AttributeError()

    for block in blocks:
        handles.append(block.register_forward_pre_hook(_qwen3_pre_hook, with_kwargs=True))
        handles.append(block.register_forward_hook(_qwen3_post_hook))

    return handles


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


def quantize_model(model_id: str, bits: int, dataset: str, device: str = "cuda", use_qwen3_hooks: bool = False):
    print(f"\nQuantizing {model_id} at {bits} bits (dataset={dataset})")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    base = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    hooks = []
    if use_qwen3_hooks:
        hooks = attach_qwen3_hooks(base)

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=dataset,
    )
    quantizer.quantize_model(base, tok)

    for h in hooks:
        h.remove()

    out_dir = f"{RESULTS_DIR}/{safe_name(model_id)}-{bits}bit-{dataset}"

    sanitize_generation_config(base)

    os.makedirs(out_dir, exist_ok=True)
    base.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    save_model(
        base,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    print(f"Saved {model_id}-{bits}bit-{dataset} -> {out_dir}")

    del quantizer
    del base
    del tok
    if device in {"cuda", "auto"}:
        torch.cuda.empty_cache()

    return out_dir


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for b in BITS:
        print(f"Quantizing Qwen3 {QWEN3_ID} at {b} bits")
        quantize_model(QWEN3_ID, b, CALIBRATION_DATASET, device="cuda", use_qwen3_hooks=True)