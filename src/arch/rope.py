
# https://github.com/aju22/RoPE-PyTorch/blob/main/RoPE.ipynb

# modified by AS

# import torch.nn as nn
# import torch


# class RotaryPositionalEmbeddings(nn.Module):

#   def __init__(self, d: int, base: int = 10_000):

#     super().__init__()
#     self.base = base
#     self.d = d
#     self.cos_cached = None
#     self.sin_cached = None

#   def _build_cache(self, x: torch.Tensor):

#     if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
#       return

#     seq_len = x.shape[0]

#     theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

#     seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]

#     idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

#     idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

#     _idx = (slice(None),) + (None,) * (x.ndim - 2) + (slice(None),)
#     self.cos_cached = idx_theta2.cos()[_idx] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
#     self.sin_cached = idx_theta2.sin()[_idx] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

#   def _neg_half(self, x: torch.Tensor):

#     d_2 = self.d // 2 #

#     return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]


#   def forward(self, x: torch.Tensor):
#     self._build_cache(x)
#     neg_half_x = self._neg_half(x)
#     x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
#     return x_rope


import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, max_seq_len: int, device: torch.device):
        # Dynamically expand the cache if the requested position exceeds current cache size
        if self.cos_cached is not None and max_seq_len <= self.cos_cached.shape[0]:
            return

        # Theta calculation: 1 / 10,000^(2i/d)
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2).float() / self.d)
        ).to(device)

        # Generate sequence indices up to the max required length
        seq_idx = torch.arange(max_seq_len, device=device).float()

        # Outer product to get (max_seq_len, d/2)
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Duplicate columns to match the full dimension d -> (max_seq_len, d)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache the raw 2D tensors (seq_len, d)
        self.cos_cached = idx_theta2.cos()
        self.sin_cached = idx_theta2.sin()

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:self.d], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None):
        """Args:

        x: Input tensor of shape (seq_len, batch_size, ..., d) or (batch_size,
        seq_len, ..., d) position_ids: Target position indices matching the
        sequence dimensions, e.g., (seq_len, batch_size)
        """
        # Ensure cache is large enough for the highest position index in this batch
        if position_ids is None:
          max_pos = int(x.shape[1])
        else:
          max_pos = int(position_ids.max().item()) + 1
        self._build_cache(max_pos, x.device)

        # Index into the 2D cache using the dynamic position_ids
        # Resulting shape: (*position_ids.shape, d)
        if position_ids is None:
          cos = self.cos_cached[:max_pos]
          sin = self.sin_cached[:max_pos]
        else:
          cos = self.cos_cached[position_ids]
          sin = self.sin_cached[position_ids]

        x_rope = x.clone()
        # print(x.shape, sin.shape, cos.shape, self._neg_half(x).shape)
        x_rope[...,:self.d] = (x[...,:self.d] * cos[None,...,:self.d]) + (self._neg_half(x)[...,:self.d] * sin[None,...,:self.d]) 
        return x_rope
