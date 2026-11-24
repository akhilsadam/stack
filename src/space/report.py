"""Structured reporting of what the stack is currently doing.

An op's value is built from a few parts: the mux chooses the weights, the
stages after it (the affine, ...) transform them, and the gate mixes the
result against the skip.  Each part describes itself with one small report:

    MuxReport     -- which channels are on, with what weight, with what prob
    AffineReport  -- a scale and a shift applied on top of the weights
    GateReport    -- how the op mixes its own value against the skip; it is
                     the last stage and scales everything before it by 1 - skip

and an `OpReport` is simply the *list* of them, in the order they act:

    OpReport(index, reports=[MuxReport, AffineReport, ..., GateReport])

Nothing is chained or nested.  The values everything else reads -- the
diagram, the text rendering, the plots -- are the *combination* of the list:
`op.final` folds the reports in order into one `Combined`.  To see the parts
instead of the total, iterate `op.reports`.

Every report is a plain description; a stage report only has to implement
`apply(state) -> state`, i.e. what it does to the running combination.

Shape convention
----------------
Every numeric payload of a report is a tensor whose FIRST axis is the channel
axis and whose REMAINING axes are spatial:

    (K,)        one value per channel, uniform over space   (today's muxes)
    (K, H, W)   one value per channel per cell              (a 2D mux)

Nothing in the rendering or plotting code needs to know which case it is
looking at: text rendering collapses a spatial field to mean +- std, the
plotting helpers hand the field to matplotlib unchanged.  A 2D mux therefore
only has to implement `channel_report` -- the whole reporting path already
works for it.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "Combined",
    "StageReport",
    "MuxReport",
    "AffineReport",
    "GateReport",
    "OpReport",
    "select_channels",
    "format_value",
    "render_mux",
    "render_gate",
    "render_op",
    "plot_op",
    "plot_stack",
]


# --------------------------------------------------------------------------
# the combination: what a list of stage reports adds up to
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Combined:
    """The op's operation after every stage has been applied.

    ids           channel (token) ids, in channel order
    weights       (K, *spatial) the weight each channel is mixed with
    probs         (K, *spatial) probability of each channel, or None when the
                  mux has no probabilistic meaning.  Not required to be a
                  normalized distribution (the ternary mux reports the
                  probability of keeping the continuous value).
    hard_weights  (K, *spatial) the coefficient the hard (eval) forward mixes,
                  or None when it makes no difference there
    mask          (K, *spatial) bool -- where the channel contributes at all
    offset        the constant added on top of the weighted mix, or None
    """

    ids: List[int] = field(default_factory=list)
    weights: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    probs: Optional[torch.Tensor] = None
    hard_weights: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    offset: Optional[float] = None

    @property
    def has_probs(self) -> bool:
        return self.probs is not None

    @property
    def is_spatial(self) -> bool:
        return self.weights.dim() > 1

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return tuple(self.weights.shape[1:])

    @property
    def size(self) -> int:
        return len(self.ids)


# --------------------------------------------------------------------------
# the stage reports
# --------------------------------------------------------------------------


class StageReport:
    """One entry of an op's report list."""

    def apply(self, state: Combined) -> Combined:
        """What this stage does to the running combination."""
        raise NotImplementedError


@dataclass(eq=False)
class MuxReport(StageReport):
    """The mux: it *chooses* the weights, so it starts the combination.

    ids           channel (token) ids, in channel order.  Dead channels are
                  dropped by a 1D mux; a 2D mux keeps them and marks the dead
                  cells in `mask`.
    weights       (K, *spatial) the mux's own weights.  For a signed weight
                  mux this is the parameter; for a probabilistic mux it is
                  the (normalized) mixing coefficient.  Later stages scale it.
    probs         (K, *spatial) probability of each channel, or None
    hard_weights  (K, *spatial) the coefficient the hard (eval) forward mixes
    mask          (K, *spatial) bool -- where the channel contributes at all
    """

    ids: List[int]
    weights: torch.Tensor
    probs: Optional[torch.Tensor] = None
    hard_weights: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    label: str = ""

    def apply(self, state: Combined) -> Combined:
        return Combined(
            ids=self.ids,
            weights=self.weights,
            probs=self.probs,
            hard_weights=self.hard_weights,
            mask=self.mask,
        )


@dataclass(eq=False)
class AffineReport(StageReport):
    """``out = scale * mixed + offset``, applied on top of what came before."""

    scale: Union[float, torch.Tensor] = 1.0
    offset: Optional[Union[float, torch.Tensor]] = None
    label: str = ""

    def apply(self, state: Combined) -> Combined:
        scale = self.scale[state.ids] if torch.is_tensor(self.scale) and self.scale.dim() > 0 else self.scale
        hard = state.hard_weights

        offset = state.offset
        if self.offset is not None:
            off = self.offset[state.ids] if torch.is_tensor(self.offset) and self.offset.dim() > 0 else self.offset
            off = (state.weights * off).sum(dim=0) if torch.is_tensor(off) and off.dim() > 0 else off
            offset = off if offset is None else offset + off

        return replace(
            state,
            weights=state.weights * scale,
            hard_weights=None if hard is None else hard * scale,
            offset=offset,
        )

@dataclass(eq=False)
class GateReport(StageReport):
    """How one op mixes its own value against the skip.

    The gate is the last stage of an op.  The forward returns
    ``(1 - skip) * value + skip * previous``, so the gate scales everything
    the stages before it produced (the weights *and* the offset) by
    ``1 - skip``.  There is no separate "keep": it is always 1 - skip.  The
    skip itself is the coefficient of the previous value on the stack.

    values       (G, *spatial) the coefficients as the gate reports them: one
                 entry (the op's own value, so skip = 1 - values[0]) or, for a
                 softmax gate, two (own value, skip).
    hard_values  (G, *spatial) the same for the hard (eval) forward, or None
                 when it is the same as `values`.
    """

    values: torch.Tensor
    hard_values: Optional[torch.Tensor] = None
    label: str = ""

    @staticmethod
    def _skip_of(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] > 1:
            return values[1]
        return 1.0 - values[0]

    @property
    def is_spatial(self) -> bool:
        return self.values.dim() > 1

    @property
    def skip(self) -> torch.Tensor:
        """Coefficient of the previous value."""
        return self._skip_of(self.values)

    @property
    def skip_hard(self) -> torch.Tensor:
        if self.hard_values is None:
            return self.skip
        return self._skip_of(self.hard_values)

    def apply(self, state: Combined) -> Combined:
        keep = 1.0 - self.skip
        keep_hard = 1.0 - self.skip_hard
        hard = state.hard_weights
        offset = state.offset
        return replace(
            state,
            weights=state.weights * keep,
            hard_weights=None if hard is None else hard * keep_hard,
            offset=None if offset is None else offset * keep,
        )


# --------------------------------------------------------------------------
# the op report: a list of stage reports
# --------------------------------------------------------------------------


def _via_final(name: str) -> property:
    """A read-only shortcut to one attribute of the combination."""
    return property(lambda self: getattr(self.final, name))


@dataclass(eq=False)
class OpReport:
    """The full report of one op of the stack.

    reports   the stage reports in the order they act: the mux first, then
              the stages after it (the affine, ...), then the gate last --
              it scales everything before it by ``1 - skip``
    """

    index: int
    reports: List[StageReport] = field(default_factory=list)

    @classmethod
    def collect(cls, index: int, reports: Sequence[Optional[StageReport]]) -> "OpReport":
        """Build from whatever the stages returned; stages with nothing to say (None) are skipped."""
        return cls(index=index, reports=[r for r in reports if r is not None])

    @property
    def final(self) -> Combined:
        """All the reports folded together, in order: the op's complete operation."""
        state = Combined()
        for report in self.reports:
            state = report.apply(state)
        return state

    @property
    def gate(self) -> Optional[GateReport]:
        return next((r for r in self.reports if isinstance(r, GateReport)), None)

    # shortcuts to the combination
    ids = _via_final("ids")
    weights = _via_final("weights")
    hard_weights = _via_final("hard_weights")
    probs = _via_final("probs")
    mask = _via_final("mask")
    offset = _via_final("offset")
    has_probs = _via_final("has_probs")
    is_spatial = _via_final("is_spatial")
    spatial_shape = _via_final("spatial_shape")

    @property
    def skip(self) -> float:
        """Scalar summary of the gate's skip coefficient (0.0 without a gate)."""
        gate = self.gate
        return 0.0 if gate is None else scalar_value(gate.skip)


def scalar_value(t) -> float:
    """Collapse a scalar-or-field to a single number (mean over the field)."""
    if t is None:
        return 0.0
    if not torch.is_tensor(t):
        return float(t)
    t = t.detach().to(torch.float32).reshape(-1)
    if t.numel() == 0:
        return 0.0
    return float(t.mean())


def select_channels(
    weights: torch.Tensor,
    alive: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
) -> Tuple[List[int], torch.Tensor, Optional[torch.Tensor]]:
    """Pick the channels that are actually doing something.

    `weights` is (n, *spatial).  `alive` marks where a channel contributes at
    all and may be given per channel (n,), per cell (*spatial) or already
    shaped like the weights.  A channel is kept when it is alive and its
    weight exceeds `threshold` *somewhere* -- so a 2D mux keeps a channel that
    is dead in part of the domain and zeroes it there.

    Returns (ids, weights of the kept channels, their liveness mask).
    """
    w = weights
    mask = None

    if alive is not None:
        a = torch.as_tensor(alive).to(torch.bool)
        if a.dim() == 1 and a.shape[0] == w.shape[0]:
            # per channel -> broadcast over the spatial axes
            a = a.reshape(a.shape[0], *([1] * (w.dim() - 1)))
        elif a.shape == tuple(w.shape[1:]):
            # per cell -> broadcast over the channel axis
            a = a.reshape(*([1] * (w.dim() - a.dim())), *a.shape)
        mask = a.expand_as(w)
        w = w * mask

    keep = w.reshape(w.shape[0], -1).abs().amax(dim=1) > threshold
    ids = torch.where(keep)[0]

    return ids.tolist(), w[ids], (mask[ids] if mask is not None else None)


# --------------------------------------------------------------------------
# text rendering
# --------------------------------------------------------------------------


def format_value(value, fmt: str = "{:.3f}", std_fmt: str = "{:.3f}") -> str:
    """Short text for a scalar or for a spatial field.

    A field is summarised as its mean and its standard deviation -- the
    number a reader wants ("on average this channel is worth 0.4, and it
    varies by 0.1 across the domain"), rather than the two extremes.
    """
    if value is None:
        return ""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    v = value.detach().to(torch.float32).reshape(-1)
    if v.numel() == 0:
        return "-"
    if v.numel() == 1:
        return fmt.format(float(v[0]))
    mean = float(v.mean())
    std = float(v.std(unbiased=False))
    return "{}±{}".format(fmt.format(mean), std_fmt.format(std))


def _same(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a.detach(), b.detach()))


def render_mux(op: Combined, name_fn: Callable[[int], str]) -> str:
    """`token(w=+0.010)` / `token(w=-0.010,p=0.120)`, one entry per channel."""
    parts = []
    for j, token_id in enumerate(op.ids):
        chunk = "{}(w={})".format(name_fn(token_id), format_value(op.weights[j], "{:+.3f}"))
        probs = op.probs
        if probs is not None and not _same(probs, op.weights):
            chunk += ",p={}".format(format_value(probs[j]))
        parts.append(chunk)

    if abs(scalar_value(op.offset)) > 1e-12:
        # the affine's shift is a constant on top of the weighted mix; the
        # diagram shows it as the channel "1", so the text does too
        parts.append("1(w={})".format(format_value(op.offset, "{:+.3f}")))
    return " ".join(parts)


def render_gate(gate: GateReport) -> str:
    """The branch coefficients; `0.731` for a sigmoid gate, `0.30/0.70` for a softmax one."""
    values = gate.values.detach().to(torch.float32)
    values = values.reshape(values.shape[0], -1)
    return "/".join(format_value(v) for v in values)


def render_op(op: OpReport, name_fn: Callable[[int], str]) -> str:
    return render_mux(op.final, name_fn)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def _quantities(stage: Combined, cmap_signed: str, cmap_prob: str):
    """The (label, values, cmap, is_signed) panels one op needs."""
    quantities = [("w", stage.weights, cmap_signed, True)]
    if stage.has_probs and not _same(stage.probs, stage.weights):
        quantities.append(("p", stage.probs, cmap_prob, False))
    return quantities


def _n_panels(stage: Combined, quantities) -> int:
    if stage.is_spatial:
        return stage.weights.shape[0] * len(quantities)
    return len(quantities)

def _draw_op_panels(fig, axs, op: OpReport, name_fn, quantities) -> None:
    """Draw one op's panels onto the given axes (left to right)."""
    from matplotlib.colors import TwoSlopeNorm

    stage = op.final
    col = 0
    for label, values, cmap, signed in quantities:
        v = values.detach().to(torch.float32).cpu()
        if stage.is_spatial:
            vmin = float(v.min())
            vmax = float(v.max())
            if signed:
                lim = max(abs(vmin), abs(vmax))
                norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            else:
                norm = None
            for j, token_id in enumerate(stage.ids):
                ax = axs[col]
                im = ax.imshow(v[j].numpy(), cmap=cmap, norm=norm, origin="lower")
                ax.set_title(
                    "op {} {} {}".format(op.index, name_fn(token_id), label),
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)
                col += 1
        else:
            ax = axs[col]
            names = [name_fn(t) for t in stage.ids]
            y = list(range(len(names)))
            vals = v.reshape(v.shape[0], -1).mean(dim=1)
            colors = (
                ["#b2182b" if x > 0 else "#2166ac" for x in vals]
                if signed
                else ["#2166ac"] * len(vals)
            )
            ax.barh(y, vals.tolist(), color=colors, height=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=8)
            ax.axvline(0.0, color="#888888", linewidth=0.8)
            ax.set_title("op {} {}".format(op.index, label), fontsize=9)
            ax.grid(axis="x", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            col += 1


def plot_op(
    op: OpReport,
    name_fn: Callable[[int], str],
    cmap_signed: str = "RdBu_r",
    cmap_prob: str = "Blues",
):
    """Small multiples of one op: weights (signed) and probabilities.

    A 1D mux gives a bar per channel; a spatial mux gives one heatmap per
    channel.  Weights and probabilities never share an axis -- they are two
    measures, so they are two panels.
    """
    from matplotlib import pyplot as plt

    quantities = _quantities(op.final, cmap_signed, cmap_prob)
    ncols = _n_panels(op.final, quantities)

    fig, axs = plt.subplots(
        1, ncols, figsize=(3.0 * ncols, 3.0), squeeze=False
    )
    _draw_op_panels(fig, axs[0], op, name_fn, quantities)
    fig.tight_layout()
    return fig


def plot_stack(
    reports: Sequence[OpReport],
    name_fn: Callable[[int], str],
    path: Optional[str] = None,
    cmap_signed: str = "RdBu_r",
    cmap_prob: str = "Blues",
):
    """One row per op: the whole stack's weights (and probabilities).

    A stack whose muxes are spatial needs one heatmap per (op, channel), so
    the grid is (ops) x (channels x quantities) -- a table of small multiples.
    A stack whose muxes are 1D instead gets one bar chart per op.
    """
    from matplotlib import pyplot as plt

    if not reports:
        raise ValueError("nothing to plot")

    # every op gets its own row; the columns are as many as the widest op
    # needs (a spatial op needs one panel per channel and quantity)
    per_op = []
    for op in reports:
        quantities = _quantities(op.final, cmap_signed, cmap_prob)
        per_op.append((op, quantities, _n_panels(op.final, quantities)))

    nrows = len(per_op)
    ncols = max(n for _, _, n in per_op)

    # a spatial row draws heatmaps and needs to be taller than a 1D bar row
    heights = [3.4 if op.final.is_spatial else 2.2 for op, _, _ in per_op]

    fig = plt.figure(figsize=(3.5 * ncols, sum(heights)))
    gs = fig.add_gridspec(
        nrows, ncols, height_ratios=heights, hspace=0.45, wspace=0.3
    )
    axs = [[fig.add_subplot(gs[row, col]) for col in range(ncols)] for row in range(nrows)]

    for row, (op, quantities, n) in enumerate(per_op):
        _draw_op_panels(fig, axs[row], op, name_fn, quantities)
        for col in range(n, ncols):  # hide the unused tail of a short row
            axs[row][col].set_visible(False)

    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig