import torch
import torch.nn as nn
from einops import rearrange

class StackRoPE(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Ensure head_dim can be cleanly split for our 2 topological dimensions
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE split."
        self.split_dim = self.head_dim // 2
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        # Sinusoidal frequencies for a single split half
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.split_dim, 2).float() / self.split_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rotation(self, coord_slice):
        # coord_slice: (B, L) -> specific coordinate feature
        angles = torch.einsum("bl,d->bld", coord_slice.float(), self.inv_freq)
        angles = rearrange(angles, "b l d -> b 1 l (d 2)")
        return angles.sin(), angles.cos()

    def _apply_half_rope(self, x_half, sin, cos):
        # Applies standard rotary rotation to a slice of the head dimension
        x_rot = torch.stack([-x_half[..., 1::2], x_half[..., 0::2]], dim=-1).flatten(-2)
        return (x_half * cos) + (x_rot * sin)

    def forward(self, x, topology):
        # x: (B, L, D)
        # tree_topology: (B, L, 2) -> [stack_slot_index, arity_branch_index]
        B, L, D = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=self.num_heads, d=self.head_dim)
        
        # Split Q and K into two halves for Multi-D rotation
        q_stack, q_arity = q[..., :self.split_dim], q[..., self.split_dim:]
        k_stack, k_arity = k[..., :self.split_dim], k[..., self.split_dim:]
        
        # Compute angles for each dimension separately
        sin_s, cos_s = self._compute_rotation(topology[..., 0]) # Stack Slot Axis
        sin_a, cos_a = self._compute_rotation(topology[..., 1]) # Arity Branch Axis
        
        # Rotate each half independently
        q_stack = self._apply_half_rope(q_stack, sin_s, cos_s)
        q_arity = self._apply_half_rope(q_arity, sin_a, cos_a)
        
        k_stack = self._apply_half_rope(k_stack, sin_s, cos_s)
        k_arity = self._apply_half_rope(k_arity, sin_a, cos_a)
        
        # Recombine the rotated halves
        q = torch.cat([q_stack, q_arity], dim=-1)
        k = torch.cat([k_stack, k_arity], dim=-1)
        
        # Fast attention execution
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)