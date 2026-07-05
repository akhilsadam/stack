# https://arxiv.org/pdf/2310.01749
# https://arxiv.org/pdf/2507.15343

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

class RPN_AE(nn.Module):
    """Pool sequence embeddings and project to contrastive space."""

    def __init__(self, embedder, TOKEN_TO_ID, ID_TO_ARITY, seq_len=100, embed_dim: int=32, proj_dim: int=64, num_heads=4):
        super().__init__()
        self.embedder = embedder
        
        # self.proj = nn.Sequential(
        #     SelfAttention(embed_dim, num_heads=num_heads),
        #     LinearLayer(embed_dim),
        #     SelfAttention(embed_dim, num_heads=num_heads),
        #     LinearLayer(embed_dim),
        #     SelfAttention(embed_dim, num_heads=num_heads),
        #     LinearLayer(embed_dim),
        #     SelfAttention(embed_dim, num_heads=num_heads),
        #     LinearLayer(embed_dim),
        #     nn.Linear(embed_dim, proj_dim),
        # )
        
        # self.unproj = nn.Sequential(
        #     SelfAttention(proj_dim + embed_dim, num_heads=num_heads),
        #     LinearLayer(proj_dim + embed_dim),
        #     SelfAttention(proj_dim + embed_dim, num_heads=num_heads),
        #     LinearLayer(proj_dim + embed_dim),
        #     SelfAttention(proj_dim + embed_dim, num_heads=num_heads),
        #     LinearLayer(proj_dim + embed_dim),
        #     nn.Linear(proj_dim + embed_dim, embed_dim),
        # )
        
        
        self.lift = nn.Linear(embed_dim, proj_dim)
        E = proj_dim
        
        self.combiner = nn.Sequential(
            nn.Linear(3 * E, 3 * E),
            nn.GELU(),
            nn.Linear(3 * E, E),
            nn.GELU(),
            nn.Linear(E, E),
        )
                
        E = proj_dim + embed_dim
        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(E, nhead=4, dim_feedforward=2*E, batch_first=True),
            num_layers=4,
        )    
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.pad_token_id = TOKEN_TO_ID["__pad__"]
        
    def zero(self):
        id_ = torch.full((1, 1), self.pad_token_id, dtype=torch.long, device=self.pe_fwd.device)
        amp_ = torch.ones(1, 1, dtype=torch.float32, device=id_.device)
        pad = self.embedder(id_, amp_)
        return pad

    def collect(self, rep, ids):
        # ids is B L
        # rep is B L E
        B, L, E = rep.shape
        results = []

        for b in range(B):
            stack = []
            for i in range(L):
                arity = self.id_to_arity.get(int(ids[b, i]), 0)
                if arity == -1:
                    continue
                args = [stack.pop() for _ in range(arity)][::-1]
                args += [torch.zeros(E, device=rep.device)] * (2 - arity)
                out = self.combiner(
                    torch.cat([rep[b, i], args[0], args[1]], dim=0)[None,:,:]  # B, 3 * E
                )
                stack.append(out)
                
            results.append(stack[-1])  # single root

        return torch.stack(results)  # B E
    
    def forward(self, rep: torch.Tensor, ids = None) -> torch.Tensor:
        zero = self.zero()
        rep = rep - zero
        x = self.lift(rep)
        x = self.collect(x, ids)
        return x # B, proj_dim

    def denoise(self, noisy, z_clean):
        # z_clean is B P, noisy is B L E
        z_exp = z_clean.unsqueeze(1).expand(-1, noisy.shape[1], -1)  # B L E
        x = torch.cat([z_exp, noisy], dim=-1)                        # B L 2E
        x = self.denoiser(x)                                          # B L 2E
        return self.decoder(x)                                        # B L E  (decoder is Linear(2E, E))

    def reverse(self, rep, pooled):

        x = self.denoise(rep, pooled)
        zero = self.zero()
        x = x + zero
        
        return x  # B, seq_len, embed_dim