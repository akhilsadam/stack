"""The stages an op's value passes through.

An op's value is more than its mux: a few transformations are applied to it
before it lands on the operand stack.  Each transformation is a *stage*.

A stage does two things:
  * it transforms the value (``forward``), and
  * it can describe what it did (``report``).

A stage's report is independent of the others: it is one entry of the op's
report list (see :class:`space.report.OpReport`), and the op's complete
operation is the combination of that list.  A stage that has nothing to say
returns None and simply does not appear in the list.
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

    def report(self, **kwargs):
        """A :class:`space.report.StageReport` describing this stage.

        Returns None when the stage has nothing to say -- it is then left
        out of the op's report list.
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