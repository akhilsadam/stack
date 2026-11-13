"""The stages an op's value passes through.

An op's value is more than its mux: a few transformations are applied to it
before it lands on the operand stack.  Each transformation is a *stage*.

A stage does two things:
  * it transforms the value (``forward``), and
  * it can describe what it did (``report``).

The reports form a chain: the mux reports the weights it chose, and every
stage after it transforms that report.  The report of the last stage is
therefore the op's *complete* operation -- which is what the diagram shows.
"""

import torch
import torch.nn as nn

__all__ = ["Stage", "Blank", "Identity"]


class Stage(nn.Module):
    """One stage of an op's computation."""

    def __init__(self):
        super().__init__()

    def forward(self, _in, *args):
        """The value this stage produces.  Identity by default."""
        return _in

    def report(self, prev, **kwargs):
        """What this stage did to the report of the stage before it.

        Returns the new report, or None when the stage has nothing to say
        -- then the report is left as it was.
        """
        return None


class Blank(Stage):
    """A stage that changes nothing.

    This is what ``preprocess`` and ``postprocess`` are: the hooks are
    there and they are empty.  A composition that wants one of them to do
    something can put a stage that does in its place.
    """

    def __init__(self):
        super().__init__()


class Identity:
    """The blank composition: ``preprocess`` and ``postprocess`` do nothing.

    This is a mixin rather than a stage: it is mixed into the op itself,
    next to the mux and the gate, and it only fills in the two hooks.
    """

    def __init__(self):
        self.preprocess = Blank()
        self.postprocess = Blank()
