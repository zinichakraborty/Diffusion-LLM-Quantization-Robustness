"""
Pseudocode:

1. Calculate number of layers and avoid quantizing embed, norm, and lm_head
2. Sort by sensitivites and use greedy algo to get assignments
"""

import json
import torch
from transformers import AutoTokenizer, AutoModel
from math import floor


MODEL_ID = "Salesforce/CoDA-v0-Instruct"


def load_coda_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.float16,
    )

    return model, tokenizer


def split_integer(N, splits):
    total = sum(splits)
    if total <= 0:
        raise ValueError("Sum of decimals must be positive")

    raw = [N * w for w in splits]
    base = [floor(x) for x in raw]

    remainder = N - sum(base)

    frac_with_idx = sorted(
        ((raw[i] - base[i], i) for i in range(len(splits))), reverse=True
    )

    for k in range(remainder):
        _, idx = frac_with_idx[k]
        base[idx] += 1

    return base


def load_sensitivities_json(sensitivities_json_path):
    with open(sensitivities_json_path) as f:
        data = json.load(f)

    del data["model.embed_tokens"]
    del data["model.norm"]
    del data["lm_head.weight"]

    return data

def parse_splits(splits_str):
    splits = splits_str.strip().split('/')
    if len(splits) != 3:
        raise ValueError(f"{splits} is not length 3")
    splits = [int(split) / 100 for split in splits]
    if sum(splits) != 1:
        raise ValueError(f"{splits} does not sum to 1")
    return splits

def calculate_bit_assignments_per_block(sensitivities, splits):
    num_layers = len(sensitivities)
    num_layers_per_bit = split_integer(num_layers, splits)
    num_16_bit_layers = num_layers_per_bit[0]
    num_8_bit_layers = num_layers_per_bit[1]

    sorted_sensitivities = dict(sorted(
        sensitivities.items(), key=lambda item: item[1], reverse=True
    ))
    sorted_layers = list(sorted_sensitivities.keys())

    layers_16_bit = sorted_layers[:num_16_bit_layers]
    layers_8_bit = sorted_layers[
        num_16_bit_layers : num_16_bit_layers + num_8_bit_layers
    ]
    layers_4_bit = sorted_layers[num_16_bit_layers + num_8_bit_layers :]
    
    assignments = {}
    for layer in layers_16_bit:
        assignments[layer] = 16
    for layer in layers_8_bit:
        assignments[layer] = 8
    for layer in layers_4_bit:
        assignments[layer] = 4
    
    return assignments