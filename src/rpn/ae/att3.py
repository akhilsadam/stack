

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
        return self.linear(attn_output)  # Residual connection

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
    def __init__(self, seq_len, embed_dim, channel_dim):
        super().__init__()

        # Token mixing (mix across sequence)
        self.token_att = SelfAttention(embed_dim, num_heads=8)

        # Channel mixing (mix across embedding)
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, embed_dim),
        )

    def forward(self, x):  # (B, L, E)
        # --- Token mixing ---
        x = x + self.token_att(x)
        # --- Channel mixing ---
        x = x + self.channel_mlp(x)
        return x

class RPN_AE(nn.Module):
    """Pool sequence embeddings and project to contrastive space."""

    def __init__(self, embedder, TOKEN_TO_ID, ID_TO_ARITY, seq_len=100, embed_dim: int=32, proj_dim: int=64, num_heads=4):
        super().__init__()
        self.embedder = embedder
        
        token_dim = embed_dim + proj_dim
        
        self.proj = nn.Sequential(
            MixerBlock(seq_len, embed_dim, proj_dim),
            MixerBlock(seq_len, embed_dim, proj_dim),
            MixerBlock(seq_len, embed_dim, proj_dim),
            MixerBlock(seq_len, embed_dim, proj_dim),
            MixerBlock(seq_len, embed_dim, proj_dim),
            MixerBlock(seq_len, embed_dim, proj_dim),
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pool_attn  = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.squash     = nn.Linear(embed_dim, proj_dim)

        # self.squash = nn.Sequential(
        #     nn.Flatten(-2,-1),
        #     nn.Linear(seq_len * embed_dim, proj_dim),
        # )
        # self.lift = nn.Sequential(
        #     nn.Linear(proj_dim, seq_len * proj_dim),
        #     nn.Unflatten(-1, (seq_len, proj_dim)),
        # )

        self.unproj = nn.Sequential(
            MixerBlock(seq_len, token_dim, token_dim * 2),
            MixerBlock(seq_len, token_dim, token_dim * 2),
            MixerBlock(seq_len, token_dim, token_dim * 2),
            MixerBlock(seq_len, token_dim, token_dim * 2),
            MixerBlock(seq_len, token_dim, token_dim * 2),
            MixerBlock(seq_len, token_dim, token_dim * 2),
            nn.Linear(token_dim, embed_dim),
        )      
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self.pe = nn.Embedding(seq_len, proj_dim)

        self.pad_token_id = TOKEN_TO_ID["__pad__"]
        
        
    def zero(self, x):
        id_ = torch.full((1, 1), self.pad_token_id, dtype=torch.long, device=x.device)
        amp_ = torch.ones(1, 1, dtype=torch.float32, device=id_.device)
        pad = self.embedder(id_, amp_)
        return pad


    def pool(self, x):  # x is B L E
        q = self.pool_query.expand(x.shape[0], -1, -1)  # B 1 E
        out, _ = self.pool_attn(q, x, x)                # B 1 E
        return self.squash(out.squeeze(1))               # B proj_dim
    
    def lift(self, z):  # z is B proj_dim
        pos = torch.arange(self.seq_len, device=z.device)
        pe  = self.pe(pos)[None].expand(z.shape[0], -1, -1)   # B L proj_dim
        z_  = z[:, None, :].expand(-1, self.seq_len, -1)       # B L proj_dim
        return z_ + pe                                          # B L proj_dim
    
    def forward(self, rep: torch.Tensor, ids = None) -> torch.Tensor:
        zero = self.zero(rep)
        rep = rep - zero
        x = rep
        x = self.proj(x)
        # return x
        # return x.sum(dim=1)  # B, proj_dim
        return self.pool(x)
    
    def reverse(self, rep, pooled):
        x = torch.cat([
            rep, 
            self.lift(pooled)
        ], dim=-1)
        
        x = self.unproj(x)
        
        zero = self.zero(x)
        x = x + zero
        
        return x  # B, seq_len, embed_dim