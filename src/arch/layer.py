import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionalEmbeddings
from .cope import DualAxisCoPERotaryEmbedding as CoPE

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

        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len
        )
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def forward(self, x):  # (B, L, D) -> (B, L, D)
        # --- Multi-Head Self-Attention (Token Mixing) ---
        norm_x = self.norm1(x)
        L = x.shape[1]

        attn_mask = self.attn_mask[:L,:L]
        # Query, Key, Value are all norm_x for self-attention
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False, is_causal=True)
        x = x + attn_out

        # --- Channel MLP (Channel Mixing) ---
        x = x + self.channel_mlp(self.norm2(x))

        return x

class FastAttentionBlockCoPE(nn.Module):
    """Shape-preserving (B, seq_len, dim) Multi-Head Self-Attention block.
    
    Uses PyTorch's native scaled dot-product attention (FlashAttention backends 
    where available) for high throughput.
    """
    def __init__(self, seq_len=None, dim=512, num_heads=8, tok_dim=32, channel_hidden=None):
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

        self.gate_proj = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.SiLU(),
            nn.Linear(channel_hidden, 1)
        )

        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len
        )
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.rope = CoPE(tok_dim//2, tok_dim//2, base=seq_len)

        self.register_buffer('pos_seq', torch.arange(seq_len), persistent=False)  # [0, 1, ..., L-1]

    def forward(self, x):  # (B, L, D) -> (B, L, D)
        # --- Multi-Head Self-Attention (Token Mixing) ---
        norm_x = self.norm1(x)
        L = x.shape[1]

        attn_mask = self.attn_mask[:L,:L]

        # gate_proj is a small Linear layer projecting hidden_dim -> 1
        delta = self.gate_proj(norm_x).squeeze(-1)  
        learned_depth = torch.cumsum(delta, dim=-1) 

        # Pass both to the dual-axis RoPE
        pos_seq = self.pos_seq[None, :L].expand(x.shape[0], -1)
        q_rope = self.rope(norm_x, pos_seq=pos_seq, pos_ctx=learned_depth)
        k_rope = self.rope(norm_x, pos_seq=pos_seq, pos_ctx=learned_depth)

        # --- Multi-Head Self-Attention (Token Mixing) ---
        attn_out, _ = self.attn(q_rope, k_rope, norm_x, attn_mask=attn_mask, need_weights=False, is_causal=True)
        x = x + attn_out

        # --- Channel MLP (Channel Mixing) ---
        x = x + self.channel_mlp(self.norm2(x))

        return x

class FastAttentionBlockCoPEv2(nn.Module):
    """Shape-preserving (B, seq_len, dim) Multi-Head Self-Attention block.
    
    Uses PyTorch's native scaled dot-product attention (FlashAttention backends 
    where available) for high throughput.
    """
    def __init__(self, seq_len=None, dim=512, num_heads=8, tok_dim=32, channel_hidden=None):
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
            nn.Linear(channel_hidden, channel_hidden),
            nn.SiLU(),
            nn.Linear(channel_hidden, dim)
        )

        # Gating heads that look at vectors to decide stack actions on-the-fly
        # Tanh allows values between -1 and 1 (or scaled further via weights)
        self.stack_gate = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.SiLU(),
            nn.Linear(channel_hidden, 1),
        )

        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len
        )
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.rope = CoPE(tok_dim//2, tok_dim//2, base=seq_len**2)

        self.register_buffer('pos_seq', torch.arange(seq_len), persistent=False)  # [0, 1, ..., L-1]

    def forward(self, x):  # (B, L, D) -> (B, L, D)
        # --- Multi-Head Self-Attention (Token Mixing) ---
        norm_x = self.norm1(x)
        L = x.shape[1]

        attn_mask = self.attn_mask[:L,:L]

        # Pass both to the dual-axis RoPE
        pos_seq = self.pos_seq[None, :L].expand(x.shape[0], -1)

        stack_gate = self.stack_gate(norm_x).squeeze(-1)  # (B, L) 
        stack = torch.cumsum(stack_gate, dim=-1) # (B, L)
        
        # --- Multi-Head Self-Attention (Token Mixing) ---
        q_rope = self.rope(norm_x, pos_seq=pos_seq, pos_ctx=stack)
        k_rope = self.rope(norm_x, pos_seq=pos_seq, pos_ctx=stack)
        attn_out, _ = self.attn(q_rope, k_rope, norm_x, attn_mask=attn_mask, need_weights=False, is_causal=True)
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

    def __init__(self, seq_len, dim, tok_dim, depth=3, num_heads=32, token_hidden=None, channel_hidden=None, out_dim=None):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        # self.blocks = nn.Sequential(*[
        #     MixerBlock2(seq_len, dim, token_hidden, channel_hidden)
        #     for _ in range(depth)
        # ])

        # self.rope = RotaryPositionalEmbeddings(d=dim-1) # IGNORE physical dims!

        blocks = []
        for _ in range(depth):
            blocks.extend([
                # RotaryPositionalEmbeddings(d=dim-1, base=seq_len), # very important it seems, esp every layer too!
                # FastAttentionBlock(seq_len=seq_len, dim=dim, num_heads=num_heads, channel_hidden=channel_hidden)

                # FastAttentionBlockCoPE(seq_len=seq_len, dim=dim, num_heads=num_heads, tok_dim=tok_dim, channel_hidden=channel_hidden)

                FastAttentionBlockCoPEv2(seq_len=seq_len, dim=dim, num_heads=num_heads, tok_dim=tok_dim, channel_hidden=channel_hidden)
            ])

        if out_dim is not None:
            blocks.append(
                nn.Linear(dim, out_dim)
            )
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, pos=None):  # (B, seq_len, dim) -> (B, seq_len, dim)
        # x = self.rope(x, pos)
        y = self.blocks(x)
        return y
