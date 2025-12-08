import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

from accelerate import init_empty_weights
from optimum.gptq import load_quantized_model

# ---- args ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "gptq"], default="base")
    p.add_argument("--model-name", type=str, default="Salesforce/CoDA-v0-Instruct",
                   help="HF ID of base model")
    p.add_argument("--gptq-model-dir", type=str,
                   help="Directory with GPTQ-quantized model (required in gptq mode)")
    p.add_argument("--torch-dtype", type=str,
                   choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn-impl", type=str,
                   choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()

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
    # if mask.dtype == torch.bool:
    #     add = torch.zeros_like(mask, dtype=target_dtype)
    #     add = add.masked_fill(~mask, float("-inf"))
    #     return add
    # return mask.to(target_dtype)

def load_base_model(args, torch_dtype, attn_impl):
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
        device_map=args.device,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
        config=cfg if cfg is not None else None,
        **({} if cfg is not None else extra),
    )
    return model


def load_gptq_model(args, torch_dtype, attn_impl):
    if args.gptq_model_dir is None:
        raise ValueError("--gptq-model-dir is required when --mode gptq")

    gptq_dir = Path(args.gptq_model_dir)

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if attn_impl != "auto":
        cfg.attn_implementation = attn_impl

    with init_empty_weights():
        empty = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            config=cfg,
            attn_implementation=args.attn_impl,
        )

    model = load_quantized_model(
        empty,
        save_folder=str(gptq_dir),
        device_map=args.device,       # or "auto", but then use it for base too
    )
    
    return model

def test_generation(model, tokenizer, prompt):
    # print("\n===== Testing Generation =====")
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
    if hasattr(model, "diffusion_generate"):
        print("===== generating with diffusion_generate =====")
        y = model.diffusion_generate(input_ids, num_steps=16, max_length=256)
    elif hasattr(model, "sample"):
        print("===== generating with model sample =====")
        try:
            y = model.sample(input_ids, steps=16, max_tokens=256)
            print("Generated sequence shape:", tuple(y.shape))
        except Exception as e:
            print("Error in sample:", e)
    else:
        print("===== Model methods:", [m for m in dir(model) if not m.startswith("_")])
        
    print(tokenizer.decode(y[0], skip_special_tokens=True))


def main():
    args = parse_args()

    # dtype & backend (shared)
    req_dtype = get_torch_dtype(args.torch_dtype)
    _force_safe_sdpa_on_pre_ampere()
    torch_dtype = _pick_dtype_for_device(req_dtype)  # from your base script
    attn_impl = _coerce_attn_impl(args.attn_impl)

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # load model
    if args.mode == "base":
        print(f"Loading BASE model: {args.model_name}")
        model = load_base_model(args, torch_dtype, attn_impl)
    else:
        print(f"Loading GPTQ model from: {args.gptq_model_dir}")
        model = load_gptq_model(args, torch_dtype, attn_impl)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    prompt = "Write a Python function to calculate fibonacci numbers"

    test_generation(model, tokenizer, prompt)


if __name__ == "__main__":
    main()
