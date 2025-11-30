# quantize_coda_and_qwen3_gptq.py
# pip install --upgrade "optimum>=1.21.0" transformers datasets safetensors

import os
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from optimum.gptq import GPTQQuantizer
from safetensors.torch import save_model

BITS = [2, 3, 4, 8]

CODA_ID = "Salesforce/CoDA-v0-Instruct"
QWEN3_ID = "Qwen/Qwen3-1.7B"

CALIBRATION_DATASET = "wikitext2"
RESULTS_DIR = "gptq_quantized_models_4"


def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")


# =========================
# CoDA-specific hook logic
# =========================

# --- pre-hook: fix star-unpacked tensors, handle [T,H] -> [1,T,H], dedupe kwargs ---
def _coda_pre_hook(module, args, kwargs):
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
def _coda_post_hook(module, args, output):
    if getattr(module, "_coda_squeezed", False) and isinstance(output, torch.Tensor) and output.dim() == 3:
        return output.squeeze(0)                 # back to [T, H]
    return output


def attach_coda_hooks(coda_model):
    handles = []
    for block in coda_model.model.layers:
        handles.append(block.register_forward_pre_hook(_coda_pre_hook, with_kwargs=True))
        handles.append(block.register_forward_hook(_coda_post_hook))
    return handles


def sanitize_generation_config(model):
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return
    gc = model.generation_config

    # If sampling is off, unset temperature (or just set a sane default).
    if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
        try:
            # Best: unset so it won't be serialized
            gc.temperature = None
        except Exception:
            # Fallback: set to a standard default
            gc.temperature = 1.0

    # Make sure it validates; if it still complains, fall back to a clean config.
    try:
        gc.validate()
    except Exception:
        model.generation_config = GenerationConfig()  # minimal, valid config


def quantize_model(model_id: str, bits: int, dataset: str, device="cuda", use_coda_hooks: bool = False):
    print(f"\n=== Quantizing {model_id} at {bits} bits (dataset={dataset}) ===")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    # Load base in bf16 on the given device (no quantization yet)
    base = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # Attach temporary hooks ONLY for CoDA
    hooks = []
    if use_coda_hooks:
        print("Attaching CoDA hooks for packed [T,H] handling…")
        hooks = attach_coda_hooks(base)

    # Configure and run Optimum's GPTQ quantizer programmatically
    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=dataset,  # e.g. "wikitext2" or a list[str]
    )
    quantizer.quantize_model(base, tok)  # quantizes in-place

    # Remove hooks once quantization is done
    for h in hooks:
        h.remove()

    out_dir = f"{RESULTS_DIR}/{safe_name(model_id)}-{bits}bit-{dataset}"

    # ---- make generation_config valid so saving doesn't error ----
    sanitize_generation_config(base)

    os.makedirs(out_dir, exist_ok=True)
    base.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    save_model(
        base,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},   # <-- required for HF loaders
    )

    print(f"Saved {model_id}-{bits}bit-{dataset} -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # for b in BITS:
    #     print("--------------------------------")
    #     print(f"Quantizing CoDA {CODA_ID} at {b} bits…")
    #     quantize_model(CODA_ID, b, CALIBRATION_DATASET, device="cuda", use_coda_hooks=True)
    #     print("--------------------------------")

    for b in BITS:
        print("--------------------------------")
        print(f"Quantizing Qwen3 {QWEN3_ID} at {b} bits…")
        # Qwen3 uses standard HF transformer layout; no CoDA hooks needed
        quantize_model(QWEN3_ID, b, CALIBRATION_DATASET, device="cuda", use_coda_hooks=False)
        print("--------------------------------")
