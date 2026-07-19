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
            # nn.GELU(),
            Sine(),
            nn.Linear(token_hidden, seq_len),
        )

        # Channel mixing (mix across the feature dimension)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            # nn.GELU(),
            Sine(),
            nn.Linear(channel_hidden, dim),
        )

        # add initialize to small values
        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, mean=0.0, std=0.01)
    #             nn.init.normal_(m.bias, mean=0.0, std=0.01)


    def forward(self, x):  # (B, L, D)
        # --- Token mixing ---
        y = x.transpose(1, 2)          # (B, D, L)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)          # (B, L, D)
        x = (x + y)                      # residual

        # --- Channel mixing ---
        x = (x + self.channel_mlp(x))

        return x


class MLP(nn.Module):
    """MLP-Mixer style sequence-to-sequence processor.

    A small stack of :class:`MixerBlock` s that maps ``(B, seq_len, dim)`` to a
    tensor of the same shape. Used by the normalizing flow (:mod:`arch.flow`) as
    the ``f_z`` / ``f_n`` coupling maps, which require a shape-preserving
    sequence processor.
    """

    def __init__(self, seq_len, dim, depth=3, token_hidden=None, channel_hidden=None):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        self.blocks = nn.Sequential(*[
            MixerBlock(seq_len, dim, token_hidden, channel_hidden)
            for _ in range(depth)
        ])

    def forward(self, x):  # (B, seq_len, dim) -> (B, seq_len, dim)
        return self.blocks(x)
