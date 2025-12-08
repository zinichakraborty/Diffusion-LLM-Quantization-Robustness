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

    if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
        try:
            gc.temperature = None
        except Exception:
            gc.temperature = 1.0

    try:
        gc.validate()
    except Exception:
        model.generation_config = GenerationConfig()

def quantize_tensor_per_group_symmetric(
    tensor: torch.Tensor,
    num_bits: int,
    group_size: int = 128,
) -> torch.Tensor:
    
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

        group_slice = tensor_2d[:, start:end]

        max_val = group_slice.abs().amax(dim=-1, keepdim=True)
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
    if num_bits >= 16:
        return

    with torch.no_grad():
        w = linear.weight.data
        qw = quantize_tensor_per_group_symmetric(w, num_bits=num_bits, group_size=group_size)
        linear.weight.data.copy_(qw)


def quantize_block(
    block: nn.Module,
    num_bits: int,
    group_size: int = 128,
):
    if num_bits >= 16:
        return

    for name, submodule in block.named_modules():
        if isinstance(submodule, nn.Linear):
            quantize_linear_layer(submodule, num_bits=num_bits, group_size=group_size)

def quantize_model(
    model: nn.Module,
    bit_assignments: Dict[str, int],
    group_size: int = 128,
):
    
    supported_bits = {4, 8, 16}
    for name, bits in bit_assignments.items():
        if bits not in supported_bits:
            raise ValueError()

    for module_name, module in model.named_modules():
        if module_name in bit_assignments:
            bits = bit_assignments[module_name]
            if bits < 16:
                print(f"Quantizing {module_name} to {bits}-bit (group_size={group_size})")
                quantize_block(module, num_bits=bits, group_size=group_size)
            else:
                print(f"Keeping {module_name} in 16-bit precision")

    return model


def save_model(
    model: nn.Module,
    tokenizer: Any,
    save_dir: Path,
    bit_assignments: Dict[str, int],
    group_size: int = 128,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sanitize_generation_config(model)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "quantization_method": "per-group-symmetric-simulated",
        "group_size": group_size,
        "bit_assignments": bit_assignments,
    }
    with open(save_dir / "quantization_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved quantized model to {save_dir}")