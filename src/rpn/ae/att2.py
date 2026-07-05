

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rope_freqs(seq_len, dim, base=10000):
    """
    Returns angles (theta) for RoPE.

    shape: (seq_len, dim // 2)
    """
    assert dim % 2 == 0, "RoPE requires even dimension"

    # Compute inverse frequencies
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    # դիր: positions
    positions = torch.arange(seq_len)

    # Outer product: (L, dim//2)
    freqs = torch.outer(positions, inv_freq)
    return freqs


def apply_rope(x, freqs):
    """
    x: (B, L, E)
    freqs: (L, E//2)
    """
    x1 = x[..., ::2]  # even dims
    x2 = x[..., 1::2] # odd dims

    cos = freqs.cos()[None, :, :]  # (1, L, E//2)
    sin = freqs.sin()[None, :, :]

    # Apply rotation
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rotated.flatten(-2)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, freqs, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("freqs", freqs)
        
        nn.init.xavier_uniform_(self.linear.weight, gain=0.0001)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        q = apply_rope(x, self.freqs)
        k = apply_rope(x, self.freqs)
        v = x # 
        attn_output, _ = self.mha(q, k, v)
        return self.linear(attn_output) + x  # Residual connection

class LinearLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.linear(x) + x

class RPN_AE(nn.Module):
    """Pool sequence embeddings and project to contrastive space."""

    def __init__(self, embedder, TOKEN_TO_ID, ID_TO_ARITY, seq_len=100, embed_dim: int=32, proj_dim: int=64, num_heads=4):
        super().__init__()
        self.embedder = embedder
        token_dim = embed_dim + proj_dim
        self.register_buffer("freqs", get_rope_freqs(seq_len, token_dim))
        
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            SelfAttention(token_dim, self.freqs, num_heads=num_heads),
            LinearLayer(token_dim),
            SelfAttention(token_dim, self.freqs, num_heads=num_heads),
            nn.Linear(token_dim, proj_dim),
        )
        
        self.unproj = nn.Sequential(
            SelfAttention(token_dim, self.freqs, num_heads=num_heads),
            LinearLayer(token_dim),
            SelfAttention(token_dim, self.freqs, num_heads=num_heads),
            LinearLayer(token_dim),
            nn.Linear(token_dim, embed_dim),
        )
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self.pad_token_id = TOKEN_TO_ID["__pad__"]
        
    def zero(self):
        id_ = torch.full((1, 1), self.pad_token_id, dtype=torch.long, device=self.freqs.device)
        amp_ = torch.ones(1, 1, dtype=torch.float32, device=id_.device)
        pad = self.embedder(id_, amp_)
        return pad

    def forward(self, rep: torch.Tensor, ids = None) -> torch.Tensor:
        zero = self.zero()
        rep = rep - zero
        x = rep
        x = self.proj(x)
        # return x
        return x.sum(dim=1)  # B, proj_dim
    
    def reverse(self, rep, pooled):
        x = torch.cat([
            rep,
            pooled[:,None,:].expand(-1, self.seq_len, self.proj_dim)
        ], dim=-1)
        
        x = self.unproj(x)
        
        zero = self.zero()
        x = x + zero
        
        return x  # B, seq_len, embed_dim