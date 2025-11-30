# pip install --upgrade "optimum>=1.21.0" "transformers>=4.45.0" "datasets>=2.19.0"

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from optimum.gptq import GPTQQuantizer
from safetensors.torch import save_model

torch.backends.cuda.matmul.fp32_precision = "high"   # or "medium", "highest"
torch.backends.cudnn.conv.fp32_precision = "high"
BITS = [2, 3, 4, 8]
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"


def safe_name(hub_id: str) -> str:
    return hub_id.replace("/", "__")


# ---------- Hooks to fix Optimum's 2D hidden_states ----------

def _qwen_pre_hook(module, args, kwargs):
    """
    Optimum sometimes replays blocks with hidden_states of shape [T, H].
    Qwen3 expects [B, T, H]. If we see 2D, add a fake batch dim.
    """
    args = list(args)
    if args and isinstance(args[0], torch.Tensor) and args[0].dim() == 2:
        # [T, H] -> [1, T, H]
        args[0] = args[0].unsqueeze(0)
        module._qwen_squeezed = True
    else:
        module._qwen_squeezed = False

    return tuple(args), kwargs


def _qwen_post_hook(module, args, output):
    """
    If we added a fake batch dim and the block returns [1, T, H],
    squeeze back to [T, H] so Optimum's bookkeeping still matches.
    """
    if getattr(module, "_qwen_squeezed", False):
        if isinstance(output, torch.Tensor) and output.dim() == 3:
            # [1, T, H] -> [T, H]
            return output.squeeze(0)
    return output


def attach_qwen_hooks(model):
    """
    Attach hooks to each decoder block (model.layers[...] for Qwen3).
    Returns list of handle objects so we can remove them later.
    """
    handles = []
    # Qwen3ForCausalLM -> model: Qwen3Model -> layers: ModuleList[Qwen3DecoderLayer]
    for block in model.model.layers:
        handles.append(block.register_forward_pre_hook(_qwen_pre_hook, with_kwargs=True))
        handles.append(block.register_forward_hook(_qwen_post_hook))
    return handles


# ---------- Misc helpers ----------

def sanitize_generation_config(model):
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return

    gc = model.generation_config

    # If do_sample=False and temperature=0, Transformers sometimes complains when saving.
    if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
        try:
            gc.temperature = None
        except Exception:
            gc.temperature = 1.0

    try:
        gc.validate()
    except Exception:
        model.generation_config = GenerationConfig()


def ensure_pad_token(model, tok):
    """
    Qwen variants often don't have pad_token set. Safest: mirror eos_token.
    """
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id


def quantize_qwen(
    model_id: str,
    bits: int,
    dataset: str,
    device: str = "cuda",
    results_dir: str = "gptq_quantized_models_qwen3",
):
    # Optional but recommended to silence TF32 warnings the "new" way
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,   # Qwen tokenizer: slow is usually safer
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,         # "cuda", "auto", or dict
        dtype=torch.bfloat16,      # <- new API: dtype= instead of torch_dtype=
        low_cpu_mem_usage=True,
    )

    ensure_pad_token(base, tokenizer)

    # Attach hooks so Optimum doesn't break rotary embeddings with 2D inputs
    hook_handles = attach_qwen_hooks(base)

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=dataset,           # e.g. "wikitext2" or list[str]
    )

    # This quantizes `base` in-place
    quantizer.quantize_model(base, tokenizer)

    # Remove hooks right after quantization
    for h in hook_handles:
        h.remove()

    out_dir = f"{results_dir}/{safe_name(model_id)}-{bits}bit-{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    sanitize_generation_config(base)

    # Save config and tokenizer
    base.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save quantized weights as safetensors
    save_model(
        base,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    print(f"Saved {model_id}-{bits}bit-{dataset} -> {out_dir}")
    return out_dir


# ---------- CLI wrapper ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--bits", nargs="+", type=int, default=BITS)
    parser.add_argument("--calib-dataset", type=str, default="wikitext2")
    parser.add_argument("--results-dir", type=str, default="gptq_quantized_models_qwen3")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    for b in args.bits:
        print("--------------------------------")
        print(f"Quantizing {args.model_id}-{b}bit…")
        quantize_qwen(
            model_id=args.model_id,
            bits=b,
            dataset=args.calib_dataset,
            device=args.device,
            results_dir=args.results_dir,
        )
        print("--------------------------------")
