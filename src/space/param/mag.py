"""How alive is each channel of an op?

The mux needs to know which of its channels are doing something.  For a 1D
mux that is one number per channel: how big the channel's output is,
compared to the others.  For a spatial mux it is the same number at every
point of space, so the mux can drop the dead cells but keep the channel.
"""

import torch

from space.util import eps

__all__ = ["mag", "mag_field", "MagCache", "alive_mask"]


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


class MagCache:
    """The liveness of an op's channels, cached for the report.

    ``magnitudes`` is one number per channel.  ``field`` is the same thing
    per point of space -- only measured when the mux is spatial, since it
    costs more.
    """

    def __init__(self, spatial=False):
        self.spatial = spatial
        self.magnitudes = None
        self.field = None

    def update(self, working_set):
        """Measure the channels of a working set (B, C, *spatial, n_ops)."""
        self.magnitudes = mag(working_set)
        self.field = mag_field(working_set) if self.spatial else None
        return self

    def alive(self, threshold=1e-10):
        """Where each channel contributes at all.

        (n,) for a 1D mux, (n, *spatial) for a spatial one: the field when
        there is one, so the mux can drop the dead cells and keep the
        channel.
        """
        if self.field is not None:
            return self.field > threshold
        if self.magnitudes is not None:
            return self.magnitudes > threshold
        return None


def alive_mask(kwargs, threshold=1e-10):
    """The liveness mask, from whichever cache the caller passed.

    A ``cache`` (a ``MagCache``) is the current spelling; ``cache_mag`` and
    ``cache_mag_field`` are the older ones and still work.
    """
    cache = kwargs.get('cache', None)
    if cache is not None:
        return cache.alive(threshold)
    field = kwargs.get('cache_mag_field', None)
    if field is not None:
        return field > threshold
    magnitudes = kwargs.get('cache_mag', None)
    if magnitudes is not None:
        return magnitudes > threshold
    return None
