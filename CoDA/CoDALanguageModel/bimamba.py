# CoDALanguageModel/bimamba.py
import torch
import torch.nn as nn

try:
    # Prefer Mamba2; fall back to Mamba if needed
    from mamba_ssm import Mamba2 as _Mamba
except Exception:
    from mamba_ssm import Mamba as _Mamba  # v1

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba: forward scan + backward scan + gated fusion + MLP.
    This mirrors CoDA's attention block shape: [B, L, D] -> [B, L, D].
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        fuse: str = "glu",
        dropout: float = 0.0,
        share_dir: bool = True,
        rms_eps: float = 1e-6,
        attention_bias_compat: bool = False,  # kept for interface parity
    ):
        super().__init__()
        # Two directional Mamba modules (share weights if desired)
        self.fwd = _Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = self.fwd if share_dir else _Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.fuse_mode = fuse
        self.fuse = nn.Linear(2 * d_model, 2 * d_model if fuse == "glu" else d_model)
        self.norm_in = RMSNorm(d_model, eps=rms_eps)

        self.ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        self.norm_out = RMSNorm(d_model, eps=rms_eps)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None, **_):
        """
        x: [B, L, D]
        attention_mask: (optional) CoDA may pass a 4D mask built for attention.
                        We ignore causal structure (diffusion is acausal) and only
                        try to derive a padding mask if it's easy to do.
        """
        # Normalize in
        h = self.norm_in(x)

        # Optional: derive a 1D pad mask [B, L] if a 4D mask is passed
        # CoDA creates a (1,1,L,L) causal mask multiplied by a broadcasted 1D pad.
        # For BiMamba we don't need causal masking; we just want to zero padded tokens.
        pad_mask_1d = None
        if attention_mask is not None:
            # If user provided a simple [B, L] mask, use it directly
            if attention_mask.dim() == 2:
                pad_mask_1d = attention_mask.bool()
            elif attention_mask.dim() == 4 and attention_mask.size(-1) == attention_mask.size(-2):
                # Heuristic: use the (non -inf) on the diagonal to recover valid tokens
                diag = torch.diagonal(attention_mask, dim1=-2, dim2=-1)  # [B, 1, L]
                pad_mask_1d = (diag > float("-inf")).squeeze(1)

        # Forward Mamba
        hf = self.fwd(h)  # [B, L, D]

        # Backward Mamba (reverse sequence, run, reverse back)
        hb = torch.flip(h, dims=[1])
        hb = self.bwd(hb)
        hb = torch.flip(hb, dims=[1])

        # Fuse directions
        h = torch.cat([hf, hb], dim=-1)
        if self.fuse_mode == "glu":
            u, v = self.fuse(h).chunk(2, dim=-1)
            h = u * torch.sigmoid(v)  # [B, L, D]
        else:
            h = self.fuse(h)          # [B, L, D]

        # (Optional) zero-out padded positions post fusion
        if pad_mask_1d is not None:
            h = h * pad_mask_1d.unsqueeze(-1)

        # Residual MLP + norm out
        y = x + self.drop(self.ff(h))
        return self.norm_out(y)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(orig_dtype)
