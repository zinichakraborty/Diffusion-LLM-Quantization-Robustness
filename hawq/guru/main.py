"""
Pseudo-code:

1. Get 16bit, 8bit, 4bit split
2. Sort sensitivities and assign 16/8/4 bits by sensitivity
3. Quantize layers based on bits with per-group symmetric quantization
4. Save model into model.safetensors + quantization_config.json
"""

import argparse
from pathlib import Path

from utils import (
    calculate_bit_assignments_per_block,
    load_coda_model,
    load_sensitivities_json,
    parse_splits,
)
from quant import quantize_model, save_model


def add_arguments(parser):
    parser.add_argument(
        "--sensitivities",
        type=str,
        required=True,
        help="Path to sensitivities json to use",
    )
    parser.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Splits to use for 16, 8, 4 bit. Format: X/Y/Z, sums to 100.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory where you want to save the quantized model",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Per-group size for symmetric weight quantization (GPTQ-style)",
    )


def main(args):
    splits = parse_splits(args.splits)
    sensitivities = load_sensitivities_json(Path(args.sensitivities))
    bit_assignments = calculate_bit_assignments_per_block(sensitivities, splits)

    coda_model, tokenizer = load_coda_model()

    quantize_model(
        coda_model,
        bit_assignments,
        group_size=args.group_size,
    )

    save_model(
        coda_model,
        tokenizer,
        Path(args.save_dir),
        bit_assignments=bit_assignments,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser=parser)
    args = parser.parse_args()
    main(args)
