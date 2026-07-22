import torch
import torch.nn as nn
import torch.nn.functional as F

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class MixerBlock(nn.Module):
    """One MLP-Mixer block: token mixing (across sequence) then channel mixing
    (across features), each with a residual connection. Shape-preserving on
    ``(B, seq_len, dim)``."""

    def __init__(self, seq_len, dim, token_hidden=None, channel_hidden=None):
        super().__init__()
        token_hidden = token_hidden or 4 * seq_len
        channel_hidden = channel_hidden or 4 * dim

        # Token mixing (mix across the sequence dimension)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, token_hidden),
            nn.SiLU(),
            # Sine(),
            nn.Linear(token_hidden, seq_len),
        )

        # Channel mixing (mix across the feature dimension)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.SiLU(),
            # Sine(),
            nn.Linear(channel_hidden, dim),
        )

    def forward(self, x):  # (B, L, D)
        # --- Token mixing ---
        y = x.transpose(1, 2)          # (B, D, L)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)          # (B, L, D)
        x = (x + y)                      # residual

        # --- Channel mixing ---
        x = (x + self.channel_mlp(x))

        return x

class MixerBlock2(nn.Module):
    """One MLP-Mixer block: token mixing (across sequence) then channel mixing
    (across features), each with a residual connection. Shape-preserving on
    ``(B, seq_len, dim)``."""

    def __init__(self, seq_len, dim, token_hidden=None, channel_hidden=None):
        super().__init__()
        token_hidden = token_hidden or 4 * seq_len
        channel_hidden = channel_hidden or 4 * dim

        # Token mixing (mix across the sequence dimension)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, token_hidden),
            nn.SiLU(),
            # Sine(),
            nn.Linear(token_hidden, seq_len)
        )

        # Channel mixing (mix across the feature dimension)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.SiLU(),
            # Sine(),
        )

        self.channel_mlp_2 = nn.Linear(channel_hidden, dim)

    def forward(self, x):  # (B, L, D)
        y = self.channel_mlp(x)
        y = y.transpose(1, 2)        
        y = self.token_mlp(y)
        y = y.transpose(1, 2) 
        y = (x + self.channel_mlp_2(y))        
        return y

class FastAttentionBlock(nn.Module):
    """Shape-preserving (B, seq_len, dim) Multi-Head Self-Attention block.
    
    Uses PyTorch's native scaled dot-product attention (FlashAttention backends 
    where available) for high throughput.
    """
    def __init__(self, seq_len=None, dim=512, num_heads=8, channel_hidden=None):
        super().__init__()
        # seq_len is kept in signature so it can act as a drop-in replacement,
        # but attention natively handles variable sequence lengths!
        channel_hidden = channel_hidden or 4 * dim
        
        self.norm1 = nn.LayerNorm(dim)
        # batch_first=True accepts and outputs (B, L, D) tensors
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.SiLU(),
            nn.Linear(channel_hidden, dim)
        )

    def forward(self, x):  # (B, L, D) -> (B, L, D)
        # --- Multi-Head Self-Attention (Token Mixing) ---
        norm_x = self.norm1(x)
        # Query, Key, Value are all norm_x for self-attention
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out

        # --- Channel MLP (Channel Mixing) ---
        x = x + self.channel_mlp(self.norm2(x))

        return x


class MLP(nn.Module):
    """MLP-Mixer style sequence-to-sequence processor.

    A small stack of :class:`MixerBlock` s that maps ``(B, seq_len, dim)`` to a
    tensor of the same shape. Used by the normalizing flow (:mod:`arch.flow`) as
    the ``f_z`` / ``f_n`` coupling maps, which require a shape-preserving
    sequence processor.
    """

    def __init__(self, seq_len, dim, depth=3, num_heads=8, token_hidden=None, channel_hidden=None):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        # self.blocks = nn.Sequential(*[
        #     MixerBlock2(seq_len, dim, token_hidden, channel_hidden)
        #     for _ in range(depth)
        # ])

        self.blocks = nn.Sequential(*[
            FastAttentionBlock(seq_len=seq_len, dim=dim, num_heads=num_heads, channel_hidden=channel_hidden)
            for _ in range(depth)
        ])

    def forward(self, x):  # (B, seq_len, dim) -> (B, seq_len, dim)
        return self.blocks(x)
