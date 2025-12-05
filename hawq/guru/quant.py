import json
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
from transformers import GenerationConfig

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

def quantize_tensor_per_group_symmetric(
    tensor: torch.Tensor,
    num_bits: int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Per-group symmetric *weight-only* quantization of a tensor.

    - Groups along the last dimension (like GPTQ's group_size on columns).
    - Symmetric: zero-point = 0, scale = max(|x|) / (2^{bits-1} - 1).
    - Returns a *dequantized* tensor in the original dtype, i.e., we simulate
      low-bit weights while keeping a standard fp16/fp32 model for easy use.

    This is intentionally close to GPTQ-style hyperparameters:
      bits in {2,3,4,8}, group_size=128, sym=True
    """
    if num_bits >= 16:
        return tensor
    if num_bits not in (2, 3, 4, 8):
        raise ValueError

    qmax = 2 ** (num_bits - 1) - 1
    if tensor.ndim != 2:
        raise ValueError("Expected 2D weight for Linear")

    out_features, in_features = tensor.shape
    n_groups = (in_features + group_size - 1) // group_size

    tensor_2d = tensor
    quantized = torch.empty_like(tensor_2d)

    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, in_features)

        group_slice = tensor_2d[:, start:end]  # [out_features, group_width]

        # per-row max for this group
        max_val = group_slice.abs().amax(dim=-1, keepdim=True)  # [out_features, 1]
        # Avoid division by zero
        scale = max_val / qmax
        scale[scale == 0] = 1.0

        group_int = torch.round(group_slice / scale).clamp(-qmax, qmax)
        group_q = group_int * scale

        quantized[:, start:end] = group_q

    return quantized


def quantize_linear_layer(
    linear: nn.Linear,
    num_bits: int,
    group_size: int = 128,
):
    """
    Quantize the weights of a single nn.Linear module in-place
    using per-group symmetric quantization.
    """
    if num_bits >= 16:
        return  # keep fp16 as-is

    with torch.no_grad():
        w = linear.weight.data
        qw = quantize_tensor_per_group_symmetric(w, num_bits=num_bits, group_size=group_size)
        linear.weight.data.copy_(qw)
        # GPTQ-style quantization is weight-only; keep bias in full precision.


def quantize_block(
    block: nn.Module,
    num_bits: int,
    group_size: int = 128,
):
    """
    Quantize all Linear submodules inside a Transformer block (CoDADecoderLayer).

    This matches GPTQ's true_sequential=True idea at a coarse level:
    we quantize whole blocks layerwise, using already-quantized predecessors. :contentReference[oaicite:3]{index=3}
    """
    if num_bits >= 16:
        return

    for name, submodule in block.named_modules():
        if isinstance(submodule, nn.Linear):
            quantize_linear_layer(submodule, num_bits=num_bits, group_size=group_size)


# -----------------------
# Public API: quantize + save
# -----------------------

def quantize_model(
    model: nn.Module,
    bit_assignments: Dict[str, int],
    group_size: int = 128,
):
    """
    Apply mixed-precision per-group symmetric quantization:

    - `bit_assignments` maps *module names* (e.g. "model.layers.0") -> bitwidth (4/8/16).
    - We walk model.named_modules() and for matching names, we quantize the
      Linear layers in that block.

    Example of keys we're expecting (from CoDA's architecture):
      - "model.layers.0", "model.layers.1", ..., "model.layers.27"
    """

    # Sanity check
    supported_bits = {4, 8, 16}
    for name, bits in bit_assignments.items():
        if bits not in supported_bits:
            raise ValueError(f"Unsupported bitwidth {bits} for layer {name}")

    # Walk all modules and quantize those whose name appears in bit_assignments
    for module_name, module in model.named_modules():
        if module_name in bit_assignments:
            bits = bit_assignments[module_name]
            if bits < 16:
                print(f"[quantize_model] Quantizing {module_name} to {bits}-bit (group_size={group_size})")
                quantize_block(module, num_bits=bits, group_size=group_size)
            else:
                print(f"[quantize_model] Keeping {module_name} in 16-bit precision")

    return model


def save_model(
    model: nn.Module,
    tokenizer: Any,
    save_dir: Path,
    bit_assignments: Dict[str, int],
    group_size: int = 128,
):
    """
    Save the quantized model:

    - We rely on Hugging Face `save_pretrained` with safe_serialization=True so
      you get `model.safetensors`.
    - Also store a small JSON sidecar with the quantization metadata
      (bit assignments + group size).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the model weights (and config) in HF format.
    sanitize_generation_config(model)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    # Save quantization metadata for reproducibility.
    meta = {
        "quantization_method": "per-group-symmetric-simulated",
        "group_size": group_size,
        "bit_assignments": bit_assignments,
    }
    with open(save_dir / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved quantized model to {save_dir}")