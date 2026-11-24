"""How alive is each channel of an op?

The mux needs to know which of its channels are doing something.  For a 1D
mux that is one number per channel: how big the channel's output is,
compared to the others.  For a spatial mux it is the same number at every
point of space, so the mux can drop the dead cells but keep the channel.
"""

import torch

from space.util import eps

__all__ = ["mag", "mag_field", "alive_mask"]


def mag(ws):
    """(B, C, *spatial, n_ops) -> (n_ops,): the size of each channel."""
    ws = ws.reshape(-1, ws.shape[-1])  # (BHW*ops, C)
    m = torch.linalg.norm(ws, dim=0)
    return (m / (m.mean() + eps))


def mag_field(ws):
    """The magnitude of each op's output at each point of space.

    ``ws`` is (B, C, *spatial, n_ops) and the result is
    (n_ops, *spatial): a spatial mux uses it to know where a channel is
    alive, so it can drop the dead cells but keep the channel.
    """
    n = ws.shape[-1]
    spatial = ws.shape[2:-1]
    lead = ws.dim() - 1 - len(spatial)      # batch and channel
    x = ws.movedim(-1, 0)                  # (n, B, C, *spatial)
    m = torch.linalg.vector_norm(x, dim=tuple(range(1, lead + 1)))
    return (m / (m.mean() + eps))


def alive_mask(working_set, threshold=1e-10, spatial=False):
    """Where each channel contributes at all.

    (n,) for a 1D mux, (n, *spatial) for a spatial one: the field when
    spatial is True, so the mux can drop the dead cells and keep the channel.
    """
    if working_set is None:
        return None
    if spatial:
        return mag_field(working_set) > threshold
    return mag(working_set) > threshold
