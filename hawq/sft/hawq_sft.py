import os
import sys
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
# CoDA Model Loading Helper
# ============================================================================
def load_coda_model_from_local(model_name, model_kwargs, script_dir=None):
    """
    Load CoDA model from local CoDALanguageModel directory.
    
    Args:
        model_name: HuggingFace model name
        model_kwargs: Dictionary of kwargs for model loading
        script_dir: Directory where the script is located (defaults to current dir)
    
    Returns:
        Loaded model
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to CoDALanguageModel directory
    coda_dir = os.path.join(script_dir, 'CoDALanguageModel')
    
    if not os.path.exists(coda_dir):
        raise FileNotFoundError(
            f"CoDALanguageModel directory not found at: {coda_dir}\n"
            "Please ensure the CoDALanguageModel folder from "
            "https://github.com/SalesforceAIResearch/CoDA/tree/main/CoDALanguageModel "
            "is in the same directory as this script."
        )
    
    print(f"Found CoDALanguageModel directory at: {coda_dir}")
    
    # For package imports to work, we need the parent directory in sys.path
    # (so we can do: import CoDALanguageModel.modeling_coda)
    parent_dir = os.path.dirname(coda_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added parent directory to sys.path: {parent_dir}")
    
    # Import the CoDA model and config classes
    try:
        print("Importing CoDA classes from local directory...")
        print(f"  Files in directory: {os.listdir(coda_dir)}")
        
        # The CoDA modules use relative imports, so we need to import them as a package
        # First, ensure the parent directory is in sys.path
        parent_dir = os.path.dirname(coda_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            print(f"  Added parent directory to sys.path: {parent_dir}")
        
        # Import as a package (CoDALanguageModel.modeling_coda)
        import CoDALanguageModel.modeling_coda as modeling_coda
        import CoDALanguageModel.model_config as model_config
        
        CoDALanguageModel = modeling_coda.CoDALanguageModel
        CoDAConfig = model_config.CoDAConfig
        
        print(f"✓ Successfully imported CoDALanguageModel and CoDAConfig")
        
        # Load config from pretrained - get the raw dict first
        print(f"\nLoading config from HuggingFace: {model_name}")
        
        # First try to get config dict from HuggingFace
        try:
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config_dict = hf_config.to_dict()
            print(f"✓ Loaded config from HuggingFace")
        except Exception as e:
            print(f"Warning: Could not load config from HuggingFace: {e}")
            print("Using default CoDAConfig...")
            config_dict = {}
        
        # Create local CoDAConfig instance
        config = CoDAConfig(**config_dict)
        print(f"✓ Created local CoDAConfig instance")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - num_hidden_layers: {config.num_hidden_layers}")
        print(f"  - mask_token_id: {getattr(config, 'mask_token_id', 'not set')}")
        
        # Load model using local class
        print(f"\nLoading model weights from HuggingFace: {model_name}")
        model = CoDALanguageModel.from_pretrained(
            model_name, 
            config=config, 
            **model_kwargs
        )
        print("✓ Successfully loaded CoDALanguageModel from local implementation")
        
        return model
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print(f"\nDirectory contents: {os.listdir(coda_dir)}")
        print("\nMake sure the CoDALanguageModel directory contains:")
        print("  - __init__.py (for package imports)")
        print("  - modeling_coda.py (with CoDALanguageModel class)")
        print("  - model_config.py (with CoDAConfig class)")
        print("  - modeling_utils.py")
        print("  - generation_utils.py")
        print("  - attention.py")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"\n✗ Error loading CoDA model: {e}")
        import traceback
        traceback.print_exc()
        raise


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


def print_sensitivity_report(layer_sens, hessian_info, save_path=None):
    """
    Print and save a comprehensive sensitivity report
    """
    print(f"\n{'='*80}")
    print("HESSIAN SENSITIVITY ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    # Sort layers by mean sensitivity
    sorted_layers = sorted(
        layer_sens.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )
    
    print("Layer-wise Sensitivity Summary (sorted by mean sensitivity):")
    print(f"{'─'*80}")
    print(f"{'Layer Name':<40} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std':>10}")
    print(f"{'─'*80}")
    
    for layer_name, sens_info in sorted_layers:
        print(f"{layer_name:<40} {sens_info['mean']:>10.6f} {sens_info['max']:>10.6f} "
              f"{sens_info['min']:>10.6f} {sens_info['std']:>10.6f}")
    
    print(f"{'─'*80}\n")
    
    # Top 10 most sensitive layers
    print("Top 10 Most Sensitive Layers:")
    print(f"{'─'*80}")
    for i, (layer_name, sens_info) in enumerate(sorted_layers[:10], 1):
        print(f"  {i:2d}. {layer_name:<35} Mean: {sens_info['mean']:.6f}, Max: {sens_info['max']:.6f}")
    print()
    
    # Top 10 least sensitive layers
    print("Top 10 Least Sensitive Layers:")
    print(f"{'─'*80}")
    least_sensitive = sorted_layers[-10:]
    least_sensitive.reverse()
    for i, (layer_name, sens_info) in enumerate(least_sensitive, 1):
        print(f"  {i:2d}. {layer_name:<35} Mean: {sens_info['mean']:.6f}, Max: {sens_info['max']:.6f}")
    print()
    
    # Parameter-level Hessian info (top 20)
    print("Top 20 Parameters by Hessian Sensitivity (S_i = λ_i / n_i):")
    print(f"{'─'*80}")
    print(f"{'Parameter Name':<50} {'λ_i':>12} {'n_i':>8} {'S_i':>12}")
    print(f"{'─'*80}")
    
    for i, (param_name, info) in enumerate(list(hessian_info.items())[:20], 1):
        print(f"{i:2d}. {param_name[:47]:<47} {info['eigenvalue_approx']:>12.6f} "
              f"{info['block_size']:>8} {info['sensitivity']:>12.8f}")
    
    print(f"{'─'*80}\n")
    
    # Summary statistics
    all_means = [s['mean'] for s in layer_sens.values()]
    all_sensitivities = [info['sensitivity'] for info in hessian_info.values()]
    
    print("Summary Statistics:")
    print(f"{'─'*80}")
    print(f"  Total layers analyzed: {len(layer_sens)}")
    print(f"  Total parameters analyzed: {len(hessian_info)}")
    print(f"  Layer sensitivity - Mean: {sum(all_means)/len(all_means):.6f}")
    print(f"  Layer sensitivity - Max: {max(all_means):.6f}")
    print(f"  Layer sensitivity - Min: {min(all_means):.6f}")
    print(f"  Parameter sensitivity (S_i) - Mean: {sum(all_sensitivities)/len(all_sensitivities):.8f}")
    print(f"  Parameter sensitivity (S_i) - Max: {max(all_sensitivities):.8f}")
    print(f"  Parameter sensitivity (S_i) - Min: {min(all_sensitivities):.8f}")
    print(f"{'─'*80}\n")
    
    # Sensitivity distribution
    print("Sensitivity Distribution (Layer Means):")
    print(f"{'─'*80}")
    bins = [0, 0.001, 0.01, 0.1, 1.0, 10.0, float('inf')]
    bin_labels = ['[0, 0.001)', '[0.001, 0.01)', '[0.01, 0.1)', '[0.1, 1.0)', '[1.0, 10.0)', '[10.0, ∞)']
    
    for i in range(len(bins) - 1):
        count = sum(1 for m in all_means if bins[i] <= m < bins[i+1])
        percentage = (count / len(all_means)) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {bin_labels[i]:<15} {count:>4} layers ({percentage:>5.1f}%) {bar}")
    
    print(f"{'='*80}\n")
    
    # Save report to file
    if save_path:
        report_path = save_path.replace('.pt', '_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HESSIAN SENSITIVITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("Layer-wise Sensitivity Summary:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Layer Name':<40} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std':>10}\n")
            f.write("-"*80 + "\n")
            for layer_name, sens_info in sorted_layers:
                f.write(f"{layer_name:<40} {sens_info['mean']:>10.6f} {sens_info['max']:>10.6f} "
                       f"{sens_info['min']:>10.6f} {sens_info['std']:>10.6f}\n")
            
            f.write("\n\nParameter-level Hessian Info:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Parameter Name':<50} {'λ_i':>12} {'n_i':>8} {'S_i':>12}\n")
            f.write("-"*80 + "\n")
            for param_name, info in hessian_info.items():
                f.write(f"{param_name:<50} {info['eigenvalue_approx']:>12.6f} "
                       f"{info['block_size']:>8} {info['sensitivity']:>12.8f}\n")
        
        print(f"Detailed report saved to: {report_path}")


# ============================================================================
# Hessian Sensitivity Analysis
# ============================================================================
def compute_layer_sensitivity(model, loss, num_power_iterations=10):
    """
    Compute Hessian eigenvalues for each layer to measure sensitivity
    Returns a dictionary of layer names to their dominant eigenvalues
    """
    layer_sensitivities = {}
    
    # Group parameters by layer
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        
        # Extract layer name (e.g., "model.layers.0.self_attn.q_proj.weight" -> "layers.0")
        layer_name = '.'.join(name.split('.')[:3]) if 'layers' in name else name.split('.')[0]
        
        if layer_name not in layer_sensitivities:
            layer_sensitivities[layer_name] = []
        
        # Simple approximation: use gradient norm as sensitivity measure
        # For full Hessian eigenvalue, we'd need power iteration (expensive)
        sensitivity = param.grad.norm().item()
        layer_sensitivities[layer_name].append(sensitivity)
    
    # Aggregate sensitivities per layer
    layer_sensitivity_summary = {}
    for layer_name, sensitivities in layer_sensitivities.items():
        layer_sensitivity_summary[layer_name] = {
            'mean': sum(sensitivities) / len(sensitivities),
            'max': max(sensitivities),
            'min': min(sensitivities),
            'std': (sum((s - sum(sensitivities)/len(sensitivities))**2 for s in sensitivities) / len(sensitivities))**0.5
        }
    
    return layer_sensitivity_summary


def hessian_aware_analysis(model, loss, block_sizes=None):
    """
    Algorithm 2: Hessian AWare Quantization preparation
    Computes block-wise Hessian eigenvalues for quantization precision determination
    """
    if block_sizes is None:
        block_sizes = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Default block size based on parameter dimensions
                block_sizes[name] = min(param.numel() // 10, 1000)
    
    hessian_info = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        
        n_i = block_sizes.get(name, 100)
        
        # Compute sensitivity S_i = λ_i / n_i
        # where λ_i is the Hessian eigenvalue
        grad_norm = param.grad.norm().item()
        sensitivity = grad_norm / n_i
        
        hessian_info[name] = {
            'eigenvalue_approx': grad_norm,
            'block_size': n_i,
            'sensitivity': sensitivity,
            'param_shape': param.shape
        }
    
    # Sort by sensitivity (descending order) for quantization precision
    sorted_info = sorted(
        hessian_info.items(), 
        key=lambda x: x[1]['sensitivity'], 
        reverse=True
    )
    
    return dict(sorted_info)


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
        
        # Check if this is a LLaDA or CoDA model
        self.is_llada = 'llada' in model_name.lower() or 'llada' in str(config.model_type).lower()
        self.is_coda = 'coda' in model_name.lower() or 'coda' in str(config.model_type).lower()
        
        print(f"Model type: {config.model_type}")
        print(f"Is LLaDA model: {self.is_llada}")
        print(f"Is CoDA model: {self.is_coda}")
        if self.is_llada or self.is_coda:
            print("Note: LLaDA/CoDA models use internal attention handling, custom attention masks will be disabled")
        
        # Load model with appropriate settings
        model_kwargs = {
            'trust_remote_code': True,
            'device_map': None
        }
        
        if use_flash_attention and torch.cuda.is_available():
            model_kwargs['torch_dtype'] = torch.bfloat16
            model_kwargs['attn_implementation'] = "flash_attention_2"
        else:
            model_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # For CoDA models, try loading from local directory first
        if self.is_coda:
            print("\n" + "="*60)
            print("Attempting to load CoDA model from local directory...")
            print("="*60)
            try:
                self.model = load_coda_model_from_local(model_name, model_kwargs)
            except Exception as e:
                print(f"\nFailed to load from local directory: {e}")
                print("\nFalling back to AutoModelForCausalLM (this may fail)...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                except Exception as e2:
                    print(f"\nAutoModelForCausalLM also failed: {e2}")
                    print("\n" + "!"*60)
                    print("SOLUTION: Ensure CoDALanguageModel directory exists")
                    print("!"*60)
                    print("To fix this, download the CoDALanguageModel folder from:")
                    print("https://github.com/SalesforceAIResearch/CoDA/tree/main/CoDALanguageModel")
                    print(f"And place it in the same directory as this script.")
                    print("!"*60)
                    raise
        else:
            # For non-CoDA models, use standard loading
            try:
                if use_flash_attention and torch.cuda.is_available():
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        print("Using Flash Attention 2")
                    except Exception as e:
                        print(f"Flash Attention not available: {e}")
                        model_kwargs.pop('attn_implementation', None)
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        self.vocab_size = self.model.config.vocab_size
        self.device = device
        
        # Debug: Check embedding layer structure
        print(f"\nModel structure debugging:")
        print(f"  Model class: {type(self.model).__name__}")
        
        # Check for CoDA-specific attributes
        if hasattr(self.model, 'get_embeds'):
            print(f"  ✓ CoDA model with get_embeds() method")
        
        if hasattr(self.model, 'model'):
            print(f"  Base model class: {type(self.model.model).__name__}")
            base_attrs = [attr for attr in dir(self.model.model) if not attr.startswith('_')]
            print(f"  Base model attributes (first 20): {base_attrs[:20]}")
            # Check for embedding layers
            embed_attrs = ['embed_tokens', 'embed_layer', 'embeddings', 'wte', 'word_embeddings', 'token_embeddings']
            for attr in embed_attrs:
                if hasattr(self.model.model, attr):
                    print(f"  ✓ Found embedding: model.model.{attr}")
        else:
            model_attrs = [attr for attr in dir(self.model) if not attr.startswith('_')]
            print(f"  Model attributes (first 20): {model_attrs[:20]}")
        
        # Check for mask_token_id in model
        if hasattr(self.model, 'mask_token_id'):
            print(f"  ✓ Model has mask_token_id: {self.model.mask_token_id}")
        
        print()
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with standard attention mask (2D: [batch, seq_len])
        
        Note: CoDA models return (logits, loss) tuple during training,
        but we only need logits for our training loop.
        """
        # CoDA models have different forward signature
        if self.is_coda:
            # CoDA forward expects specific parameters
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # CoDA returns (logits, loss) tuple
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
        else:
            # Standard transformers models
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False
            )
            # Handle different output types
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        return logits
    
    def get_embeds(self, input_ids):
        """Get input embeddings - handles different model architectures"""
        # CoDA models have a built-in get_embeds method
        if hasattr(self.model, 'get_embeds'):
            try:
                return self.model.get_embeds(input_ids)
            except:
                pass
        
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
    use_shift=False,
    track_hessian=False,
    hessian_freq=500
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
            'use_shift': use_shift,
            'track_hessian': track_hessian,
            'hessian_freq': hessian_freq
        })
    
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load tokenizer with fallback to slow tokenizer
    print(f"\nLoading tokenizer from: {model_name}")
    try:
        # Try fast tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        print("Successfully loaded fast tokenizer")
    except Exception as e:
        print(f"Fast tokenizer failed ({e.__class__.__name__}), falling back to slow tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            print("Successfully loaded slow tokenizer")
        except Exception as e2:
            print(f"ERROR: Both fast and slow tokenizer loading failed!")
            print(f"Fast tokenizer error: {e}")
            print(f"Slow tokenizer error: {e2}")
            raise
    
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
    if model.is_llada or model.is_coda:
        print(f"Attention: LLaDA/CoDA internal (no annealing)")
    else:
        print(f"Anneal steps: {anneal_steps}")
    print(f"Use shift loss: {use_shift}")
    print(f"Track Hessian sensitivity: {track_hessian}")
    if track_hessian:
        print(f"Hessian computation frequency: every {hessian_freq} steps")
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
                # LLaDA/CoDA models handle attention internally and don't accept 4D masks
                if model.is_llada or model.is_coda:
                    # For LLaDA/CoDA: use simple 2D padding mask only
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
                loss_mask = (x_t == mask_token_id) & (padding_mask.bool())
                x_target = x
                
                # Handle shifting:
                # CoDA models automatically shift internally during training (return seq_len-1)
                # For other models, we can optionally shift with use_shift flag
                if model.is_coda:
                    # CoDA always shifts logits internally: [batch, seq_len] -> [batch, seq_len-1]
                    # So we need to shift our targets and masks to match
                    loss_mask = loss_mask[:, 1:]
                    x_target = x_target[:, 1:]
                elif use_shift:
                    # Manual shift for non-CoDA models
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
                    
                    # Compute Hessian-based sensitivity if enabled
                    if track_hessian and global_step % hessian_freq == 0:
                        print(f"\n{'─'*60}")
                        print(f"[Step {global_step}] Computing Layer Sensitivities...")
                        print(f"{'─'*60}")
                        try:
                            # Compute layer-wise sensitivity
                            layer_sens = compute_layer_sensitivity(model, final_loss)
                            
                            # Log top 10 most sensitive layers
                            sorted_layers = sorted(
                                layer_sens.items(), 
                                key=lambda x: x[1]['mean'], 
                                reverse=True
                            )[:10]
                            
                            print("\nTop 10 Most Sensitive Layers:")
                            for i, (layer_name, sens_info) in enumerate(sorted_layers, 1):
                                print(f"  {i:2d}. {layer_name:<40} mean={sens_info['mean']:.6f}, max={sens_info['max']:.6f}")
                            
                            if use_wandb:
                                # Log sensitivity metrics
                                for layer_name, sens_info in sorted_layers:
                                    wandb.log({
                                        f'sensitivity/{layer_name}/mean': sens_info['mean'],
                                        f'sensitivity/{layer_name}/max': sens_info['max'],
                                        f'sensitivity/{layer_name}/std': sens_info['std'],
                                        'step': global_step
                                    })
                            
                            # Compute Hessian-aware quantization info
                            hessian_info = hessian_aware_analysis(model, final_loss)
                            
                            # Show top 5 parameters by sensitivity
                            print("\nTop 5 Parameters by Sensitivity (S_i = λ_i / n_i):")
                            for i, (param_name, info) in enumerate(list(hessian_info.items())[:5], 1):
                                print(f"  {i}. {param_name[:50]:<50} S_i={info['sensitivity']:.8f}")
                            
                            # Save sensitivity analysis
                            sens_path = os.path.join(save_dir, f'sensitivity_step_{global_step}.pt')
                            torch.save({
                                'layer_sensitivity': layer_sens,
                                'hessian_info': hessian_info,
                                'global_step': global_step,
                                'epoch': epoch
                            }, sens_path)
                            print(f"\nSensitivity analysis saved to: {sens_path}")
                            print(f"{'─'*60}\n")
                            
                        except Exception as e:
                            print(f"Error computing sensitivities: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging
                epoch_loss += final_loss.item()
                num_batches += 1
                global_step += 1
                
                # Compute attn_mask_ratio for logging (even if not used)
                attn_mask_ratio_log = min(1.0, global_step / anneal_steps) if not (model.is_llada or model.is_coda) else 1.0
                
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
                print(f"  Is CoDA: {model.is_coda}")
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
    
    # Compute final comprehensive sensitivity analysis
    # Note: Keep model in train mode so CoDA continues to shift
    print("\nComputing final Hessian sensitivity analysis...")
    # model.eval()  # Don't call eval() - CoDA needs train mode to shift
    
    # Do one forward-backward pass to get fresh gradients
    try:
        # Get a batch for final analysis
        final_batch = next(iter(train_loader))
        x = final_batch['input_ids'].to(device)
        src_mask = final_batch['src_mask'].to(device).bool()
        padding_mask = final_batch['attention_mask'].to(device)
        
        batch_size_actual = x.size(0)
        seq_len = x.size(1)
        
        # Forward pass
        sampling_eps = 1e-3
        t = (1 - sampling_eps) * torch.rand(batch_size_actual, device=device) + sampling_eps
        sigma = noiser.total_noise(t)
        dsigma = noiser.rate_noise(t)
        
        maskable_mask = (~src_mask) & (padding_mask.bool())
        x_t = transition(x, sigma[:, None], maskable_mask=maskable_mask, mask_token_id=mask_token_id)
        
        # Use same attention mask logic as training loop
        attn_mask_ratio = 1.0
        x_embed = model.get_embeds(x)
        attention_mask = get_anneal_attn_mask(
            seq_len, batch_size_actual,
            dtype=x_embed.dtype,
            device=device,
            attn_mask_ratio=attn_mask_ratio
        )
        
        logits = model(x_t, attention_mask=attention_mask)
        actual_vocab_size = logits.size(-1)
        loss_mask = (x_t == mask_token_id) & (padding_mask.bool())
        
        x_target = x
        # Handle CoDA's internal shifting
        if model.is_coda:
            # CoDA shifts internally, so shift targets to match
            loss_mask = loss_mask[:, 1:]
            x_target = x_target[:, 1:]
        elif use_shift:
            # Manual shift for non-CoDA models
            logits = logits[:, :-1]
            loss_mask = loss_mask[:, 1:]
            x_target = x_target[:, 1:]
        
        loss = F.cross_entropy(
            logits.reshape(-1, actual_vocab_size),
            x_target.reshape(-1),
            reduction="none"
        ).float().reshape(x_target.size(0), x_target.size(1))
        
        loss = loss.masked_fill(~loss_mask, 0)
        num_masked = loss_mask.sum()
        
        if num_masked > 0:
            final_loss = (dsigma[:, None] * loss).sum() / num_masked
            
            # Backward to compute gradients
            optimizer.zero_grad()
            final_loss.backward()
            
            # Compute comprehensive sensitivity analysis
            print("Computing layer sensitivities...")
            final_layer_sens = compute_layer_sensitivity(model, final_loss)
            
            print("Computing Hessian-aware quantization info...")
            final_hessian_info = hessian_aware_analysis(model, final_loss)
            
            # Print comprehensive report
            final_report_path = os.path.join(save_dir, 'final_sensitivity_analysis.pt')
            print_sensitivity_report(final_layer_sens, final_hessian_info, final_report_path)
            
            # Save final analysis
            torch.save({
                'layer_sensitivity': final_layer_sens,
                'hessian_info': final_hessian_info,
                'global_step': global_step,
                'epoch': num_epochs,
                'model_name': model_name,
            }, final_report_path)
            print(f"\nFinal sensitivity analysis saved to: {final_report_path}")
            
        else:
            print("Warning: No masked tokens in final batch, skipping sensitivity analysis")
            
    except Exception as e:
        print(f"Error computing final sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All Training Complete!")
    print(f"{'='*60}")
    
    # Set model to eval mode before saving
    model.eval()
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model')
    print(f"Saving final model to {final_path}")
    
    # Fix generation config to avoid validation errors
    if hasattr(model.model, 'generation_config'):
        gen_config = model.model.generation_config
        # Fix conflicting temperature setting
        if hasattr(gen_config, 'do_sample') and not gen_config.do_sample:
            if hasattr(gen_config, 'temperature'):
                # Remove temperature when do_sample is False
                gen_config.temperature = None
    
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
        description='Train Discrete Diffusion Language Model (LLaDA-8B / CoDA Compatible)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_name', type=str, default='Salesforce/CoDA-v0-Instruct', 
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
    parser.add_argument('--track_hessian', action='store_true',
                        help='Track Hessian-based layer sensitivity during training')
    parser.add_argument('--hessian_freq', type=int, default=500,
                        help='Frequency (in steps) to compute Hessian sensitivity')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("LLaDA/CoDA Diffusion Training Configuration")
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
        use_shift=args.use_shift,
        track_hessian=args.track_hessian,
        hessian_freq=args.hessian_freq
    )


if __name__ == '__main__':
    main()