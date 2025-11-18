import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import argparse
import wandb
import warnings

# ============================================================================
# Linear Noise Schedule
# ============================================================================
class LinearNoise:
    """Linear noise schedule for discrete diffusion"""
    def __init__(self):
        super().__init__()

    def rate_noise(self, t):
        """Weighting with (alpha_t)'/(1-alpha_t)"""
        return torch.reciprocal(t)

    def total_noise(self, t):
        """Returns noise level at time t (0~1)"""
        return t


# ============================================================================
# Attention Mask Functions
# ============================================================================
def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    """Create annealed attention mask that transitions from causal to full attention"""
    # Create causal mask (lower triangular)
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)
    
    # Create random mask based on annealing ratio
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)
    
    # Combine causal and random masks (logical OR)
    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    
    # Invert mask (1 -> 0, 0 -> 1) and apply large negative value for masking
    inverted_mask = 1.0 - expanded_mask.to(dtype)
    
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# ============================================================================
# Diffusion Functions
# ============================================================================
def transition(x_0, sigma, maskable_mask, mask_token_id):
    """Apply transition to mask tokens based on noise level"""
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t


# ============================================================================
# Dataset
# ============================================================================
class GSMDataset(Dataset):
    """GSM8K Dataset for diffusion training"""
    def __init__(self, tokenizer, split='train', max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load GSM8K dataset
        print(f"Loading GSM8K {split} split...")
        dataset = load_dataset("gsm8k", "main", split=split)
        self.data = []
        
        for item in dataset:
            question = item['question']
            answer = item['answer']
            # Format as "Question: ... Answer: ..."
            text = f"Question: {question}\nAnswer: {answer}"
            self.data.append(text)
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create source mask (no masking for this simple case)
        # src_mask marks positions that should NOT be corrupted (e.g., prompts)
        src_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'src_mask': src_mask
        }


# ============================================================================
# Model Wrapper
# ============================================================================
class DiffusionLM(nn.Module):
    """Wrapper around HuggingFace model for diffusion training"""
    def __init__(self, model_name, device='cuda', use_flash_attention=False):
        super().__init__()
        
        # Load config first to check model type
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Check if this is a LLaDA model
        self.is_llada = 'llada' in model_name.lower() or 'llada' in str(config.model_type).lower()
        
        print(f"Model type: {config.model_type}")
        print(f"Is LLaDA model: {self.is_llada}")
        if self.is_llada:
            print("Note: LLaDA models use internal attention handling, custom attention masks will be disabled")
        
        # Load model with appropriate settings
        if use_flash_attention and torch.cuda.is_available():
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                    device_map=None
                )
                print("Using Flash Attention 2")
            except Exception as e:
                print(f"Flash Attention not available: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    device_map=None
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map=None
            )
        
        self.vocab_size = self.model.config.vocab_size
        self.device = device
        
        # Debug: Check embedding layer structure
        print(f"\nModel structure debugging:")
        print(f"  Model class: {type(self.model).__name__}")
        if hasattr(self.model, 'model'):
            print(f"  Base model class: {type(self.model.model).__name__}")
            print(f"  Base model attributes: {[attr for attr in dir(self.model.model) if not attr.startswith('_')][:20]}")
            # Check for embedding layers
            embed_attrs = ['embed_tokens', 'embed_layer', 'embeddings', 'wte', 'word_embeddings', 'token_embeddings']
            for attr in embed_attrs:
                if hasattr(self.model.model, attr):
                    print(f"  ✓ Found embedding: model.model.{attr}")
        else:
            print(f"  Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')][:20]}")
        print()
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with standard attention mask (2D: [batch, seq_len])
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False
        )
        return outputs.logits
    
    def get_embeds(self, input_ids):
        """Get input embeddings - handles different model architectures"""
        # Try different paths for different model structures
        try:
            # Standard transformers models (GPT2, etc.)
            return self.model.get_input_embeddings()(input_ids)
        except:
            pass
        
        try:
            # Llama-style: model.model.embed_tokens
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens(input_ids)
        except:
            pass
        
        try:
            # LLaDA or other custom models: check for embed_layer, embedding, etc.
            if hasattr(self.model, 'model'):
                base_model = self.model.model
                # Try common embedding attribute names
                for attr_name in ['embed_layer', 'embeddings', 'wte', 'word_embeddings', 'token_embeddings']:
                    if hasattr(base_model, attr_name):
                        embed_layer = getattr(base_model, attr_name)
                        return embed_layer(input_ids)
        except:
            pass
        
        try:
            # Direct model access
            for attr_name in ['embed_layer', 'embeddings', 'wte', 'word_embeddings', 'token_embeddings', 'embed_tokens']:
                if hasattr(self.model, attr_name):
                    embed_layer = getattr(self.model, attr_name)
                    return embed_layer(input_ids)
        except:
            pass
        
        # Last resort: use get_input_embeddings from base model
        try:
            if hasattr(self.model, 'model'):
                return self.model.model.get_input_embeddings()(input_ids)
        except:
            pass
        
        raise AttributeError(
            f"Could not find embedding layer for model type {type(self.model).__name__}. "
            f"Model attributes: {dir(self.model.model if hasattr(self.model, 'model') else self.model)}"
        )


# ============================================================================
# Training Function
# ============================================================================
def train_diffusion_model(
    model_name='GSAI-ML/LLaDA-8B-Base',
    dataset_split='train',
    batch_size=8,
    num_epochs=3,
    learning_rate=1e-4,
    diffusion_steps=100,
    max_length=256,
    device='cuda',
    use_wandb=False,
    save_dir='./checkpoints',
    gradient_accumulation_steps=1,
    use_flash_attention=False,
    max_grad_norm=1.0,
    anneal_steps=1000,
    use_shift=False
):
    """Train diffusion language model"""
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="llada-diffusion-training", config={
            'model_name': model_name,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'diffusion_steps': diffusion_steps,
            'max_length': max_length,
            'anneal_steps': anneal_steps,
            'use_shift': use_shift
        })
    
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Check initial tokenizer state
    print(f"Initial vocab size: {len(tokenizer)}")
    print(f"Has mask_token: {tokenizer.mask_token is not None}")
    print(f"Has pad_token: {tokenizer.pad_token is not None}")
    
    # Handle padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set pad_token to eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        else:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            print(f"Added new pad_token: '<PAD>' (ID: {tokenizer.pad_token_id})")
    
    # Handle mask token - LLaDA models should have this, but check
    needs_resize = False
    original_vocab_size = len(tokenizer)
    
    if tokenizer.mask_token is None:
        print("\nWARNING: No mask_token found! Adding <MASK> token...")
        num_added = tokenizer.add_special_tokens({'mask_token': '<MASK>'})
        needs_resize = True
        print(f"Added {num_added} special token(s)")
    else:
        print(f"Found existing mask_token: '{tokenizer.mask_token}'")
    
    # Verify mask token
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Failed to get mask_token_id from tokenizer!")
    
    print(f"\n{'='*60}")
    print("Tokenizer Configuration:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Mask token: '{tokenizer.mask_token}' (ID: {mask_token_id})")
    print(f"  Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = DiffusionLM(model_name, device, use_flash_attention=use_flash_attention)
    
    # Resize token embeddings if needed
    if needs_resize and len(tokenizer) > model.vocab_size:
        print(f"\nResizing token embeddings from {model.vocab_size} to {len(tokenizer)}")
        model.model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = model.model.config.vocab_size  # Get from config after resize
        print(f"Model vocab size after resize: {model.vocab_size}")
    elif len(tokenizer) != model.vocab_size:
        print(f"\nWARNING: Tokenizer vocab size ({len(tokenizer)}) != model vocab size ({model.vocab_size})")
        print("Resizing to match tokenizer...")
        model.model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = model.model.config.vocab_size  # Get from config after resize
        print(f"Model vocab size after resize: {model.vocab_size}")
    
    # Move model to device
    print(f"Moving model to {device}...")
    model = model.to(device)
    
    # Verify output size with a dummy forward pass
    print("Verifying model output size...")
    with torch.no_grad():
        dummy_input = torch.zeros((1, 10), dtype=torch.long, device=device)
        dummy_output = model(dummy_input, attention_mask=torch.ones((1, 10), device=device))
        actual_output_size = dummy_output.size(-1)
        print(f"  Model output vocab size: {actual_output_size}")
        print(f"  Expected vocab size: {model.vocab_size}")
        print(f"  Tokenizer vocab size: {len(tokenizer)}")
        if actual_output_size != model.vocab_size:
            print(f"  WARNING: Mismatch detected! Using actual output size: {actual_output_size}")
            model.vocab_size = actual_output_size
    
    # Test embedding extraction
    print("Testing embedding extraction...")
    try:
        dummy_embeds = model.get_embeds(dummy_input)
        print(f"  ✓ Successfully extracted embeddings: {dummy_embeds.shape}")
    except Exception as e:
        print(f"  ✗ Failed to extract embeddings: {e}")
        print("  This will cause errors during training!")
        raise
    
    # Enable gradient checkpointing for memory efficiency (if supported)
    if hasattr(model.model, 'gradient_checkpointing_enable'):
        try:
            model.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing")
        except (ValueError, AttributeError) as e:
            print(f"Gradient checkpointing not supported: {e}")
            print("Continuing without gradient checkpointing...")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset = GSMDataset(tokenizer, split=dataset_split, max_length=max_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize optimizer and noise schedule
    print(f"\nInitializing optimizer (lr={learning_rate})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    noiser = LinearNoise()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    model.train()
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Total training steps: {num_epochs * len(train_loader)}")
    print(f"Diffusion steps: {diffusion_steps}")
    if model.is_llada:
        print(f"Attention: LLaDA internal (no annealing)")
    else:
        print(f"Anneal steps: {anneal_steps}")
    print(f"Use shift loss: {use_shift}")
    print(f"Mask token ID: {mask_token_id}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                x = batch['input_ids'].to(device)
                src_mask = batch['src_mask'].to(device).bool()
                padding_mask = batch['attention_mask'].to(device)
                
                batch_size_actual = x.size(0)
                seq_len = x.size(1)
                
                # Continuous time sampling
                sampling_eps = 1e-3
                t = (1 - sampling_eps) * torch.rand(batch_size_actual, device=device) + sampling_eps
                
                sigma = noiser.total_noise(t)
                dsigma = noiser.rate_noise(t)
                
                # Apply transition (masking)
                # Only mask positions that are not source and not padding
                # This ensures we never mask special tokens or padding
                maskable_mask = (~src_mask) & (padding_mask.bool())
                x_t = transition(x, sigma[:, None], maskable_mask=maskable_mask, mask_token_id=mask_token_id)
                
                # Compute attention mask
                # LLaDA models handle attention internally and don't accept 4D masks
                if model.is_llada:
                    # For LLaDA: use simple 2D padding mask only
                    attention_mask = padding_mask
                else:
                    # For other models: use annealed 4D attention mask
                    attn_mask_ratio = min(1.0, (global_step + 1) / anneal_steps)
                    x_embed = model.get_embeds(x)
                    attention_mask = get_anneal_attn_mask(
                        seq_len, batch_size_actual, 
                        dtype=x_embed.dtype, 
                        device=device, 
                        attn_mask_ratio=attn_mask_ratio
                    )
                
                # Forward pass
                logits = model(x_t, attention_mask=attention_mask)
                
                # Get actual vocab size from logits (in case of mismatch)
                actual_vocab_size = logits.size(-1)
                
                # Compute loss only on masked positions that are not padding
                # This matches the original: loss_mask = x_t == mask_token_id
                # But we also exclude padding for safety
                loss_mask = (x_t == mask_token_id) & (padding_mask.bool())
                
                # Apply shift if enabled
                x_target = x
                if use_shift:
                    logits = logits[:, :-1]
                    loss_mask = loss_mask[:, 1:]
                    x_target = x_target[:, 1:]
                
                # Ensure loss mask and target have same shape as logits
                assert logits.size(0) == x_target.size(0), f"Batch size mismatch: {logits.size(0)} vs {x_target.size(0)}"
                assert logits.size(1) == x_target.size(1), f"Seq len mismatch: {logits.size(1)} vs {x_target.size(1)}"
                
                # Check if we have any masked tokens
                num_masked = loss_mask.sum()
                if num_masked == 0:
                    if batch_idx % 100 == 0:
                        print(f"Warning: No masked tokens in batch {batch_idx}, skipping...")
                    continue
                
                loss = F.cross_entropy(
                    logits.reshape(-1, actual_vocab_size), 
                    x_target.reshape(-1), 
                    reduction="none"
                ).float().reshape(x_target.size(0), x_target.size(1))
                
                loss = loss.masked_fill(~loss_mask, 0)
                
                # Weighted loss
                final_loss = (dsigma[:, None] * loss).sum() / num_masked
                unweighted_loss = loss.sum() / num_masked
                
                # Backward pass with gradient accumulation
                scaled_loss = final_loss / gradient_accumulation_steps
                scaled_loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging
                epoch_loss += final_loss.item()
                num_batches += 1
                global_step += 1
                
                # Compute attn_mask_ratio for logging (even if not used)
                attn_mask_ratio_log = min(1.0, global_step / anneal_steps) if not model.is_llada else 1.0
                
                progress_bar.set_postfix({
                    'loss': f'{final_loss.item():.4f}',
                    'unw': f'{unweighted_loss.item():.4f}',
                    'masked': num_masked.item(),
                    'σ': f'{sigma.mean().item():.3f}',
                    'attn': f'{attn_mask_ratio_log:.2f}'
                })
                
                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'train/loss': final_loss.item(),
                        'train/unweighted_loss': unweighted_loss.item(),
                        'train/num_masked_tokens': num_masked.item(),
                        'train/avg_sigma': sigma.mean().item(),
                        'train/attn_mask_ratio': attn_mask_ratio_log,
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/epoch': epoch,
                        'step': global_step
                    })
            
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                print(f"  Batch size: {x.size() if 'x' in locals() else 'N/A'}")
                print(f"  Logits size: {logits.size() if 'logits' in locals() else 'N/A'}")
                print(f"  Model vocab size: {model.vocab_size}")
                print(f"  Mask token ID: {mask_token_id}")
                print(f"  Is LLaDA: {model.is_llada}")
                print(f"  Use shift: {use_shift}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Processed Batches: {num_batches}/{len(train_loader)}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        print(f"  Saving checkpoint to {checkpoint_path}")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': {
                'model_name': model_name,
                'vocab_size': len(tokenizer),
                'mask_token_id': mask_token_id,
                'diffusion_steps': diffusion_steps,
                'anneal_steps': anneal_steps,
                'use_shift': use_shift,
            }
        }, checkpoint_path)
        
        if use_wandb:
            wandb.log({
                'epoch/loss': avg_loss,
                'epoch/number': epoch + 1
            })
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model')
    print(f"Saving final model to {final_path}")
    model.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Done!")
    
    if use_wandb:
        wandb.finish()
    
    return model


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train Discrete Diffusion Language Model (LLaDA-8B Compatible)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Base', 
                        help='HuggingFace model name')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Batch size (use 2-4 for 8B models)')
    parser.add_argument('--num_epochs', type=int, default=3, 
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                        help='Learning rate')
    parser.add_argument('--diffusion_steps', type=int, default=100, 
                        help='Number of diffusion steps')
    parser.add_argument('--max_length', type=int, default=256, 
                        help='Max sequence length')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device (cuda/cpu)')
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Use Weights & Biases logging')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                        help='Save directory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                        help='Gradient accumulation steps')
    parser.add_argument('--use_flash_attention', action='store_true', 
                        help='Use Flash Attention 2 (requires flash-attn package)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                        help='Max gradient norm for clipping')
    parser.add_argument('--anneal_steps', type=int, default=1000,
                        help='Number of steps to anneal attention from causal to full')
    parser.add_argument('--use_shift', action='store_true',
                        help='Use shifted loss (predict next token)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("LLaDA Diffusion Training Configuration")
    print(f"{'='*60}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"{'='*60}\n")
    
    train_diffusion_model(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        max_length=args.max_length,
        device=args.device,
        use_wandb=args.use_wandb,
        save_dir=args.save_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_flash_attention=args.use_flash_attention,
        max_grad_norm=args.max_grad_norm,
        anneal_steps=args.anneal_steps,
        use_shift=args.use_shift
    )


if __name__ == '__main__':
    main()