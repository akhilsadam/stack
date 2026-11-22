"""The affine an op applies to its mixed value.

It used to live in the mux (``PostAffine``) and run after the muxing, so
the mux's report could not say what the op really did.  It is a stage of
its own now: the operator applies it, and it reports itself, so the report
shows the complete operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from space.report import AffineReport
from .stage import Stage

class MuxAffine(Stage):
    """``out = scale * mixed + shift``.

    The parametrization is the one the affine has always had: ``raw_scale *
    rate`` and ``raw_shift * rate``, starting at 1.0 and 0.0.

    As a stage it does two things.  In the forward it scales and shifts
    the value the op has mixed.  In the report it scales the weights the
    mux reported and carries the shift, so that the reported weights
    reconstruct the op's value exactly.
    """

    def __init__(self, scale=1.0, shift=0.0, rate=0.1):
        super().__init__()

        self.rate = rate
        self.raw_scale = nn.Parameter(torch.tensor(scale) / self.rate)
        self.raw_shift = nn.Parameter(torch.tensor(shift) / self.rate)

    @property
    def affine_scale(self):
        return self.raw_scale * self.rate

    @property
    def affine_shift(self):
        return self.raw_shift * self.rate

    def forward(self, _in, *args):
        return self.affine_scale * _in + self.affine_shift

    def report(self, **kwargs):
        return AffineReport(
            label=__class__.__name__,
            scale=float(self.affine_scale),
            offset=float(self.affine_shift),
        )