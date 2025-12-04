#!/usr/bin/env python
import os
import sys
import argparse
import math
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import GenerationConfig
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ============================================================================
# Simple Fake-Quantization Helpers for QAT
# ============================================================================
def fake_quantize_symmetric(x, num_bits=8, eps: float = 1e-8):
    """
    Symmetric per-tensor fake quantization (no zero-point).
    Simulates int{num_bits} quantization in float.

    For num_bits=8 -> q in [-127, 127].
    """
    if num_bits <= 1:
        return x  # degenerate, just bail

    qmax = 2 ** (num_bits - 1) - 1  # e.g. 127
    # Use absolute max for symmetric quant
    max_val = x.detach().abs().max()
    if max_val < eps:
        return x

    scale = max_val / qmax
    # Round to nearest integer and clamp to range
    x_int = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    x_q = x_int * scale
    return x_q


class QuantLinear(nn.Module):
    """
    Wraps a Linear layer with simple fake-quantization on weights and
    optionally activations for QAT.

    - Keeps original Linear weights in full precision.
    - Forwards use quantized copies of weight (and input if enabled).
    """

    def __init__(self, linear: nn.Linear, num_bits: int = 8, quantize_activations: bool = True):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("QuantLinear expects an nn.Linear module")

        self.linear = linear
        self.num_bits = num_bits
        self.quantize_activations = quantize_activations
        self.qat_enabled = True  # can be toggled at runtime

    def forward(self, x):
        if self.qat_enabled:
            # fake-quant activations if requested
            if self.quantize_activations:
                x = fake_quantize_symmetric(x, num_bits=self.num_bits)

            # fake-quant weights
            w = fake_quantize_symmetric(self.linear.weight, num_bits=self.num_bits)
            b = self.linear.bias
            return F.linear(x, w, b)
        else:
            # fall back to normal float forward
            return self.linear(x)


def wrap_linear_with_qat(module: nn.Module, num_bits: int = 8, quantize_activations: bool = True):
    """
    Recursively replace all nn.Linear submodules with QuantLinear wrappers.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child, num_bits, quantize_activations))
        else:
            wrap_linear_with_qat(child, num_bits, quantize_activations)


def set_qat_enabled(model: nn.Module, enabled: bool = True):
    """
    Turn QAT on/off globally by toggling qat_enabled flag of QuantLinear modules.
    """
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.qat_enabled = enabled


# ============================================================================
# Generation config sanitizer
# ============================================================================
def sanitize_generation_config(model):
    """
    Fix CoDA's broken GenerationConfig so save_pretrained() stops crashing.
    """
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return

    gc = model.generation_config

    # If sampling is off but temperature is 0.0, unset or fix it.
    if getattr(gc, "do_sample", False) is False and getattr(gc, "temperature", None) in (0, 0.0):
        try:
            # Prefer to unset so it isn't serialized
            gc.temperature = None
        except Exception:
            # Fallback: a sane default
            gc.temperature = 1.0

    # Make sure it validates; if not, fall back to a clean config.
    try:
        gc.validate()
    except Exception:
        model.generation_config = GenerationConfig()  # minimal, valid config


# ============================================================================
# CoDA Model Loading Helper (local implementation)
# ============================================================================
def load_coda_model_from_local(model_name, model_kwargs, script_dir=None):
    """
    Load CoDA model from local CoDALanguageModel directory.

    This expects the CoDA repo folder `CoDALanguageModel` to be in the same
    directory as this script:
        https://github.com/SalesforceAIResearch/CoDA/tree/main/CoDALanguageModel
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    coda_dir = os.path.join(script_dir, "CoDALanguageModel")
    if not os.path.exists(coda_dir):
        raise FileNotFoundError(
            f"CoDALanguageModel directory not found at: {coda_dir}\n"
            "Please ensure the CoDALanguageModel folder from "
            "https://github.com/SalesforceAIResearch/CoDA/tree/main/CoDALanguageModel "
            "is in the same directory as this script."
        )

    print(f"Found CoDALanguageModel directory at: {coda_dir}")

    parent_dir = os.path.dirname(coda_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added parent directory to sys.path: {parent_dir}")

    try:
        print("Importing CoDA classes from local directory...")
        print(f"  Files in CoDALanguageModel: {os.listdir(coda_dir)}")

        import CoDALanguageModel.modeling_coda as modeling_coda
        import CoDALanguageModel.model_config as model_config

        CoDALanguageModel = modeling_coda.CoDALanguageModel
        CoDAConfig = model_config.CoDAConfig

        print("✓ Successfully imported CoDALanguageModel and CoDAConfig")

        print(f"\nLoading base config from HuggingFace: {model_name}")
        try:
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config_dict = hf_config.to_dict()
            print("✓ Loaded config from HuggingFace")
        except Exception as e:
            print(f"Warning: Could not load config from HuggingFace: {e}")
            print("Using default CoDAConfig() with no overrides...")
            config_dict = {}

        config = CoDAConfig(**config_dict)
        print("✓ Created local CoDAConfig instance")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - num_hidden_layers: {config.num_hidden_layers}")
        print(f"  - mask_token_id: {getattr(config, 'mask_token_id', 'not set')}")

        print(f"\nLoading model weights from HuggingFace: {model_name}")
        model = CoDALanguageModel.from_pretrained(
            model_name,
            config=config,
            **model_kwargs,
        )
        print("✓ Successfully loaded CoDALanguageModel (local implementation)")
        return model

    except ImportError as e:
        print(f"\n✗ Import Error while loading CoDA: {e}")
        import traceback
        traceback.print_exc()
        raise

    except Exception as e:
        print(f"\n✗ Error loading CoDA model: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# GSM8K Dataset with src_mask for SFT
# ============================================================================
class GSMDataset(Dataset):
    """
    GSM8K dataset for diffusion SFT.

    We format each example as:
        "Question: {question}\nAnswer:"  (prompt / source)
        " {answer}"                       (target / completion)

    src_mask == True on the prompt tokens, False on the answer tokens.
    """

    def __init__(self, tokenizer, split="train", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading GSM8K '{split}' split...")
        raw = load_dataset("gsm8k", "main", split=split)
        self.examples = []

        for item in raw:
            q = item["question"]
            a = item["answer"]

            # Text formatting
            prompt_text = f"Question: {q}\nAnswer:"
            answer_text = f" {a}"

            # First encode full prompt+answer
            full_enc = tokenizer(
                prompt_text + answer_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)

            # Encode prompt alone to determine prompt length in tokens
            prompt_enc = tokenizer(
                prompt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids = prompt_enc["input_ids"].squeeze(0)
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                # Fallback: count all tokens (rare for GPT-style)
                prompt_len = (prompt_ids != -100).sum().item()
            else:
                prompt_len = (prompt_ids != pad_id).sum().item()

            src_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            src_mask[:prompt_len] = True
            # Also never treat pads as source
            if pad_id is not None:
                src_mask[input_ids == pad_id] = False

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "src_mask": src_mask,
                }
            )

        print(f"Loaded {len(self.examples)} GSM8K examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# WikiText-2 Dataset (full-token blocks, no src_mask)
# ============================================================================
class WikiText2Dataset(Dataset):
    """
    WikiText-2 dataset for diffusion pretraining.

    We:
      - Load all lines from the split (train/validation/test).
      - Concatenate them into one long stream of tokens.
      - Chunk into blocks of length `max_length`.
    """

    def __init__(self, tokenizer, split="train", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading WikiText-2 '{split}' split...")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        texts = [ex["text"] for ex in raw if ex["text"] and ex["text"].strip()]
        if not texts:
            raise ValueError(f"No non-empty texts found in WikiText-2 split '{split}'")

        joined = "\n\n".join(texts)

        # Tokenize as one long stream, then chunk
        enc = tokenizer(
            joined,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        full_ids = enc["input_ids"].squeeze(0)  # (T,)
        num_tokens = full_ids.size(0)

        num_blocks = num_tokens // max_length
        if num_blocks == 0:
            raise ValueError(
                f"Not enough tokens ({num_tokens}) to form a single block of length {max_length}"
            )

        full_ids = full_ids[: num_blocks * max_length]
        input_ids = full_ids.view(num_blocks, max_length)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        self.input_ids = input_ids
        self.attention_mask = attention_mask

        print(
            f"Constructed {num_blocks} blocks of length {max_length} "
            f"({num_tokens} tokens total) from WikiText-2 '{split}'."
        )

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            # No src_mask needed for pretraining (we'll pass src_mask=None)
        }


# ============================================================================
# Model loader wrapper
# ============================================================================
def load_coda_or_auto(model_name, model_kwargs):
    """
    Load CoDA either from local CoDALanguageModel sources (for base HF ID)
    or from a saved checkpoint directory via AutoModel.

    Here we always assume HF repo ID and try local CoDA implementation first.
    """
    print(f"Detected remote HF model id: {model_name}")
    try:
        return load_coda_model_from_local(model_name, model_kwargs)
    except Exception as e_local:
        print(f"Local CoDALanguageModel load failed: {e_local}")
        print("Falling back to AutoModel with remote code...")
        return AutoModel.from_pretrained(
            model_name,
            **model_kwargs,
        )


# ============================================================================
# Training Loop for CoDA (GSM8K SFT or WikiText-2 pretrain) + optional QAT
# ============================================================================
def train_coda_ddm_sft(
    model_name="Salesforce/CoDA-v0-Instruct",
    task="gsm8k",                # "gsm8k" or "wikitext2"
    dataset_split="train",
    batch_size=4,
    num_epochs=3,
    learning_rate=2e-5,
    max_length=512,
    device="cuda",
    use_wandb=False,
    save_dir="./checkpoints_coda",
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    use_qat=False,
    qat_bits=8,
    qat_warmup_steps=0,
):
    """
    Train a CoDA diffusion LM on either GSM8K (SFT) or WikiText-2 (pretraining).

    task == "gsm8k":
        - Uses GSMDataset.
        - Calls CoDA with training_mode="sft" and a src_mask.

    task == "wikitext2":
        - Uses WikiText2Dataset.
        - Calls CoDA with training_mode="pretrain" and src_mask=None.

    QAT:
        - If use_qat=True, wraps all nn.Linear layers with QuantLinear.
        - Fake-quantization is applied during forward passes (after optional warmup).
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    if use_wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb not installed but use_wandb=True")
        wandb.init(
            project=f"coda-ddm-{task}",
            config={
                "model_name": model_name,
                "task": task,
                "dataset_split": dataset_split,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "use_qat": use_qat,
                "qat_bits": qat_bits,
                "qat_warmup_steps": qat_warmup_steps,
            },
        )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer from: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        print("Successfully loaded fast tokenizer")
    except Exception as e:
        print(f"Fast tokenizer failed ({e.__class__.__name__}): {e}")
        print("Falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        print("Successfully loaded slow tokenizer")

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"mask_token: {tokenizer.mask_token} (id={tokenizer.mask_token_id})")
    print(f"pad_token:  {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"eos_token:  {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"bos_token:  {tokenizer.bos_token} (id={tokenizer.bos_token_id})")

    if tokenizer.mask_token_id is None:
        raise ValueError(
            "CoDA model should already have a mask_token; "
            "found mask_token_id=None. Make sure you're using a CoDA checkpoint."
        )

    # Ensure we have padding; for decoder-only models we often set pad=eos
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(
                f"Set pad_token to eos_token: '{tokenizer.eos_token}' "
                f"(id={tokenizer.eos_token_id})"
            )
        else:
            warnings.warn(
                "Tokenizer has no pad_token and no eos_token; "
                "padding behavior may be odd."
            )

    # ------------------------------------------------------------------
    # Model (CoDA)
    # ------------------------------------------------------------------
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": None,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }

    print(f"\nLoading model: {model_name}")
    model = load_coda_or_auto(model_name, model_kwargs)

    model.to(device)
    model.train()

    sanitize_generation_config(model)

    # Optionally wrap all Linear layers with QAT
    if use_qat:
        print(f"\n[QAT] Wrapping Linear layers with QuantLinear (bits={qat_bits})...")
        wrap_linear_with_qat(model, num_bits=qat_bits, quantize_activations=True)
        # Start with QAT disabled if warmup > 0
        if qat_warmup_steps > 0:
            set_qat_enabled(model, enabled=False)
            print(f"[QAT] Will enable QAT after {qat_warmup_steps} steps.")
        else:
            set_qat_enabled(model, enabled=True)
            print("[QAT] QAT enabled from step 0.")

    # Try to enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on model")
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    print(f"\nLoading dataset for task='{task}'...")
    if task == "gsm8k":
        train_dataset = GSMDataset(
            tokenizer=tokenizer,
            split=dataset_split,
            max_length=max_length,
        )
    elif task == "wikitext2":
        train_dataset = WikiText2Dataset(
            tokenizer=tokenizer,
            split=dataset_split,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unknown task '{task}'. Supported: 'gsm8k', 'wikitext2'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    global_step = 0
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("Starting CoDA Training")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Dataset split: {dataset_split}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Use QAT: {use_qat}, QAT bits: {qat_bits}, QAT warmup steps: {qat_warmup_steps}")
    print("=" * 60 + "\n")

    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 60}")
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if task == "gsm8k":
                # Supervised fine-tuning with prefix src_mask
                src_mask = batch["src_mask"].to(device)
                training_mode = "sft"
            else:
                # WikiText-2 pretraining: no src_mask, use default pretrain behavior
                src_mask = None
                training_mode = "pretrain"

            # Enable QAT after warmup if configured
            if use_qat and qat_warmup_steps > 0 and global_step == qat_warmup_steps:
                print(f"[QAT] Enabling QAT at global_step={global_step}")
                set_qat_enabled(model, enabled=True)

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    src_mask=src_mask,
                    labels=input_ids,
                    training_mode=training_mode,
                    epoch=epoch,
                )
            except TypeError:
                # Fallback if CoDA doesn't take training_mode/epoch/etc.
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    src_mask=src_mask,
                    labels=input_ids,
                )

            # Unpack (logits, loss) or HuggingFace ModelOutput
            if isinstance(outputs, tuple):
                logits, loss = outputs[0], outputs[1]
            else:
                logits = getattr(outputs, "logits", None)
                loss = getattr(outputs, "loss", None)

            if loss is None:
                raise RuntimeError(
                    "CoDA forward did not return a loss. "
                    "Make sure you're calling it in training mode with labels."
                )

            # Gradient accumulation
            loss_to_backprop = loss / gradient_accumulation_steps
            loss_to_backprop.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip grads
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{math.exp(loss.item()):.2f}"
                    if loss.item() < 20
                    else "inf",
                }
            )

            if use_wandb and (global_step % 10 == 0):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/perplexity": math.exp(loss.item())
                        if loss.item() < 20
                        else float("inf"),
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "task": task,
                        "qat/enabled": bool(use_qat and (qat_warmup_steps == 0 or global_step >= qat_warmup_steps)),
                    }
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  avg loss: {avg_loss:.4f}")
        if avg_loss < 20:
            print(f"  avg ppl:  {math.exp(avg_loss):.2f}")

        if use_wandb:
            wandb.log(
                {
                    "epoch/loss": avg_loss,
                    "epoch/perplexity": math.exp(avg_loss)
                    if avg_loss < 20
                    else float("inf"),
                    "epoch": epoch + 1,
                    "task": task,
                }
            )

        # Save a checkpoint each epoch
        ckpt_dir = os.path.join(save_dir, f"{task}_epoch_{epoch+1}")
        print(f"Saving epoch {epoch+1} checkpoint to: {ckpt_dir}")
        os.makedirs(ckpt_dir, exist_ok=True)

        sanitize_generation_config(model)
        model.save_pretrained(ckpt_dir, safe_serialization=False)
        tokenizer.save_pretrained(ckpt_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Final full save
    final_dir = os.path.join(save_dir, f"{task}_final")
    print(f"Saving final model to: {final_dir}")
    os.makedirs(final_dir, exist_ok=True)

    sanitize_generation_config(model)
    model.save_pretrained(final_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_dir)

    if use_wandb:
        wandb.finish()

    return model


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CoDA DDM Training (GSM8K SFT or WikiText-2 pretraining) with optional QAT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/CoDA-v0-Instruct",
        help="HuggingFace model name or local CoDA checkpoint path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "wikitext2"],
        help="Training task / dataset: gsm8k (SFT) or wikitext2 (pretraining)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'train', 'test', 'validation')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (CoDA is big; keep small)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_coda",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--use_qat",
        action="store_true",
        help="Enable Quantization-Aware Training (QAT) with fake-quantized Linear layers",
    )
    parser.add_argument(
        "--qat_bits",
        type=int,
        default=8,
        help="Number of bits for fake quantization in QAT (e.g., 8)",
    )
    parser.add_argument(
        "--qat_warmup_steps",
        type=int,
        default=0,
        help="Number of global steps before enabling QAT (0 = enable from step 0)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CoDA DDM Training Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60 + "\n")

    train_coda_ddm_sft(
        model_name=args.model_name,
        task=args.task,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
        use_wandb=args.use_wandb,
        save_dir=args.save_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_qat=args.use_qat,
        qat_bits=args.qat_bits,
        qat_warmup_steps=args.qat_warmup_steps,
    )


if __name__ == "__main__":
    main()
