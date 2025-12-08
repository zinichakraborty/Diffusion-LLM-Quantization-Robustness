import os
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from optimum.gptq import GPTQQuantizer
from safetensors.torch import save_model

BITS = [2, 3, 4, 8]
MODEL_ID = "Salesforce/CoDA-v0-Instruct"
CALIBRATION_DATASET = "wikitext2"
RESULTS_DIR = "gptq_quantized_models"
os.makedirs(RESULTS_DIR, exist_ok=True)

def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")

def _pre_hook(module, args, kwargs):
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
    module._coda_squeezed = squeezed

    if kwargs:
        kwargs = dict(kwargs)
        if len(args) >= 2 and "attention_mask" in kwargs:
            kwargs.pop("attention_mask")
        if len(args) >= 3 and "position_ids" in kwargs:
            kwargs.pop("position_ids")
        if len(args) >= 4 and "position_embeddings" in kwargs:
            kwargs.pop("position_embeddings")

    return args, (kwargs or {})

def _post_hook(module, args, output):
    if getattr(module, "_coda_squeezed", False) and isinstance(output, torch.Tensor) and output.dim() == 3:
        return output.squeeze(0)
    return output

def attach_hooks(coda_model):
    handles = []
    for block in coda_model.model.layers:
        handles.append(block.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        handles.append(block.register_forward_hook(_post_hook))
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


def quantize_coda(model_id: str, bits: int, dataset: str, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    base = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.float16,
    )

    hooks = attach_hooks(base)

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=dataset,
        backend="auto",
        checkpoint_format="gptq_v2",
        use_exllama=True if bits == 4 else False
    )
    quantizer.quantize_model(base, tok)

    for h in hooks:
        h.remove()

    out_dir = f"{RESULTS_DIR}/{safe_name(model_id)}-{bits}bit-{dataset}"
    
    sanitize_generation_config(base)

    quantizer.save(base, out_dir, safe_serialization=False)

    print(f"Saved {model_id}-{bits}bit-{dataset} -> {out_dir}")
    return out_dir

for b in BITS:
    print(f"Quantizing {MODEL_ID}-{b}bits")
    quantize_coda(MODEL_ID, b, CALIBRATION_DATASET, device="cuda")