

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
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


class MixerBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, token_dim, channel_dim):
        super().__init__()

        # Token mixing (mix across sequence)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, seq_len),
        )

        # Channel mixing (mix across embedding)
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, embed_dim),
        )

    def forward(self, x):  # (B, L, E)
        # --- Token mixing ---
        y = x.transpose(1, 2)         # (B, E, L)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)         # (B, L, E)
        x = x + y                     # residual

        # --- Channel mixing ---
        x = x + self.channel_mlp(x)

        return x

class RPN_AE(nn.Module):
    """Pool sequence embeddings and project to contrastive space."""

    def __init__(self, embedder, TOKEN_TO_ID, ID_TO_ARITY, seq_len=100, embed_dim: int=32, proj_dim: int=64, num_heads=4):
        super().__init__()
        self.embedder = embedder
        
        token_dim = 4 * embed_dim
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            nn.Flatten(-2,-1),
            nn.Linear(seq_len * token_dim, proj_dim),
        )
                
        self.unproj = nn.Sequential( 
            nn.Linear(proj_dim, seq_len * token_dim),
            nn.Unflatten(-1, (seq_len, token_dim)),
        )
        self.decode = nn.Sequential( 
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            MixerBlock(seq_len, token_dim, seq_len * 4, proj_dim),
            nn.Linear(token_dim, embed_dim),            
        )      
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.pe_fwd = nn.Parameter(0.01 * torch.randn(seq_len, embed_dim))
        self.pe_rev = nn.Parameter(0.01 * torch.randn(seq_len, token_dim))  # Learnable reverse positional encoding

        self.pad_token_id = TOKEN_TO_ID["__pad__"]
        
    def zero(self):
        id_ = torch.full((1, 1), self.pad_token_id, dtype=torch.long, device=self.pe_fwd.device)
        amp_ = torch.ones(1, 1, dtype=torch.float32, device=id_.device)
        pad = self.embedder(id_, amp_)
        return pad

    def forward(self, rep: torch.Tensor, ids = None) -> torch.Tensor:
        x = rep + self.pe_fwd[None,...]
        x = self.proj(x)
        return x

    
    def reverse(self, pooled):
        x = pooled
        x = self.unproj(x) + self.pe_rev[None,...]
        x = self.decode(x)
        return x