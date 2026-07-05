from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.SiLU(),
            nn.Linear(4*dim, dim),
        )

    def forward(self, x):
        return self.linear(x) + x

class RPN_GEN(nn.Module):
    """divide and generate"""

    def __init__(self, proj_dim: int=96, sem_dim: int=64, struct_dim: int=64, n_layers=3, steps=5):
        super().__init__()
        
        self.proj_dim = proj_dim
        self.sem_dim = sem_dim
        self.struct_dim = struct_dim
        latent_dim = sem_dim + struct_dim
        
        self.encode = nn.Sequential(
            nn.Linear(proj_dim, latent_dim),
            ResBlock(latent_dim),
            ResBlock(latent_dim)
        )

        self._gen = nn.Sequential(
            ResBlock(latent_dim),
            ResBlock(latent_dim),
            ResBlock(latent_dim)
        )
        
        self.decode = nn.Linear(latent_dim, proj_dim)

        self.crit = lambda x_hat, x: ((x_hat - x).pow(2).mean() / ((x - x.mean(dim=(-1),keepdim=True)).pow(2).mean() + 1e-8))
        
    
    def noise(self, x):
        shape = (*x.shape[:-1], self.struct_dim)
        struct = torch.randn(shape, device=x.device)
        x_n = torch.cat([x[...,:self.sem_dim], struct], dim=-1)
        return x_n
    
    def denoise(self, x):
        return self.decode(self._gen(x))
   
    def mix(self, x: torch.Tensor, n: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:
        return x * t + n * (1 - t)
    
    def swap(self, x, y):
        sem_x = x[...,:self.sem_dim]
        sem_y = y[...,:self.sem_dim]
        x2 = x.clone()
        y2 = y.clone()
        x2[...,:self.sem_dim] = sem_y
        y2[...,:self.sem_dim] = sem_x
        return x2, y2       

    def loss(self, _x, _x_p):
        x = _x.detach() # don't affect LLM part
        x_p = _x_p.detach()
        
        z = self.encode(x)
        z_p = self.encode(x_p)
        
        t = torch.rand(x.shape[0], device=x.device)[:, None]
        
        n = self.noise(z)
        z_n = self.mix(z, n, t)
        z_p_n = self.mix(z_p, n, t)
        
        z_n, z_p_n = self.swap(z_n, z_p_n) # swap semantic, no change expected
        
        x_hat = self.denoise(z_n)
        x_p_hat = self.denoise(z_p_n)
        
        z_hat = self.encode(x_hat)
        z_p_hat = self.encode(x_p_hat)
        
        return self.crit(x_hat, x) + self.crit(x_p_hat, x_p) \
             + self.crit(z_hat, z) + self.crit(z_p_hat, z_p), \
            self.crit(z_p[...,:self.sem_dim], z[...,:self.sem_dim])

    def semantic(self, x):
        return self.encode(x)[...,:self.sem_dim]

    def gen(self, x):
        n = self.noise(x)
        return self.denoise(n)
    
    def fm_gen(self, x):
        t = torch.rand(x.shape[0], device=x.device)[:, None]
        z = self.encode(x)    
        n = self.noise(z)
        z_n = self.mix(z, n, t)
        return self.denoise(z_n)
        