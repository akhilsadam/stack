from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN_AE(nn.Module):
    def __init__(self, embedder, TOKEN_TO_ID, ID_TO_ARITY, seq_len=100, embed_dim: int=32, proj_dim: int=64, num_heads=4):
        super().__init__()
        self.embedder     = embedder
        self.id_to_arity  = ID_TO_ARITY
        self.seq_len      = seq_len
        self.embed_dim    = embed_dim
        self.proj_dim     = proj_dim
        self.pad_token_id = TOKEN_TO_ID["__pad__"]

        self.lift = nn.Linear(embed_dim, proj_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(proj_dim, nhead=num_heads, dim_feedforward=2*proj_dim, batch_first=True),
            num_layers=4,
        )

        E = proj_dim + embed_dim
        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(E, nhead=num_heads, dim_feedforward=2*E, batch_first=True),
            num_layers=4,
        )
        self.decoder = nn.Linear(E, embed_dim)

    def zero(self):
        dev  = next(self.parameters()).device
        id_  = torch.full((1, 1), self.pad_token_id, dtype=torch.long, device=dev)
        amp_ = torch.ones(1, 1, dtype=torch.float32, device=dev)
        return self.embedder(id_, amp_)   # 1 1 embed_dim

    def forward(self, rep, ids=None):
        # rep is B L embed_dim
        zero = self.zero()
        rep  = rep - zero
        x    = self.lift(rep)          # B L proj_dim
        x    = self.encoder(x)         # B L proj_dim
        z    = x.mean(dim=1)           # B proj_dim
        return z

    def denoise(self, noisy, z_clean):
        # noisy is B L embed_dim, z_clean is B proj_dim
        z_exp = z_clean.unsqueeze(1).expand(-1, noisy.shape[1], -1)  # B L proj_dim
        x     = torch.cat([z_exp, noisy], dim=-1)                    # B L (proj_dim + embed_dim)
        x     = self.denoiser(x)                                      # B L (proj_dim + embed_dim)
        return self.decoder(x)                                        # B L embed_dim

    def reverse(self, rep, pooled):
        # rep is B L embed_dim (noisy), pooled is B proj_dim (clean z)
        zero = self.zero()
        x    = self.denoise(rep, pooled)
        x    = x + zero
        return x                       # B L embed_dim