import torch
import torch.nn as nn

class DualAxisCoPERotaryEmbedding(nn.Module):
    def __init__(self, d_seq: int, d_ctx: int, base: int = 10_000):
        """
        Args:
            d_seq: Number of dimensions for standard physical sequence RoPE.
            d_ctx: Number of dimensions for learned contextual RoPE (e.g., stack depth).
                   The remaining dimensions in the tensor will be unrotated (Partial RoPE).
            base: The frequency base for the rotary angles.
        """
        super().__init__()
        self.d_seq = d_seq
        self.d_ctx = d_ctx
        
        # Register inverse frequencies as buffers (they don't need gradients)
        self.register_buffer(
            "inv_freq_seq", 
            1.0 / (base ** (torch.arange(0, d_seq, 2).float() / d_seq))
        )
        
        self.register_buffer(
            "inv_freq_ctx", 
            1.0 / (base ** (torch.arange(0, d_ctx, 2).float() / d_ctx))
        )

    def _neg_half(self, x: torch.Tensor):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor):
        # pos shape: (batch_size, seq_len)
        # Calculate theta on the fly to allow gradient flow for fractional contextual positions
        freqs = torch.einsum("bs,d -> bsd", pos.float(), inv_freq)
        
        # Duplicate columns to match full chunk dimension: (batch, seq, d)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Automatically broadcast over heads if input is 4D (batch, seq, heads, dim)
        # by inserting unsqueezed dimensions before the last one
        for _ in range(x.ndim - emb.ndim):
            emb = emb.unsqueeze(-2)
            
        cos, sin = emb.cos(), emb.sin()
        return (x * cos) + (self._neg_half(x) * sin)

    def forward(self, x: torch.Tensor, pos_seq: torch.Tensor = None, pos_ctx: torch.Tensor = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, ..., dim)
            pos_seq: Physical sequence indices (batch, seq_len). 
                     If None, defaults to [0, 1, 2, ...].
            pos_ctx: Continuous/Learned contextual positions (batch, seq_len). 
                     If None, defaults to 0.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Default physical sequence IDs
        if pos_seq is None:
            pos_seq = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            pos_seq = pos_seq.unsqueeze(0).expand(batch_size, -1)
            
        # Default contextual sequence IDs (no contextual rotation if not provided)
        if pos_ctx is None:
            pos_ctx = torch.zeros_like(pos_seq)

        # Slice the tensor into three operational chunks
        x_seq = x[..., :self.d_seq]
        x_ctx = x[..., self.d_seq : self.d_seq + self.d_ctx]
        x_pass = x[..., self.d_seq + self.d_ctx:]

        # print(x_seq.shape, x_ctx.shape, x_pass.shape, self.inv_freq_seq.shape, self.inv_freq_ctx.shape)

        # Apply respective RoPEs
        x_seq_rope = self._apply_rope(x_seq, pos_seq, self.inv_freq_seq)
        x_ctx_rope = self._apply_rope(x_ctx, pos_ctx, self.inv_freq_ctx)

        # Re-concatenate: Sequence Axis | Context Axis | Unrotated
        return torch.cat([x_seq_rope, x_ctx_rope, x_pass], dim=-1)