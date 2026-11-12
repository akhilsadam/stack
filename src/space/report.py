"""Structured reporting of what the stack is currently doing.

The muxes and the gates are the only place where "which token, with what
weight, with what probability" lives.  That information used to leave the
model as ad-hoc tuples whose shapes differed from class to class (four
different `channel_ids` signatures, two different `gate_info` signatures),
which made it impossible to render consistently -- let alone to plot.

The dataclasses below replace those tuples with one uniform shape:

    MuxReport   -- which channels are on, with what weight, with what prob
    GateReport  -- how the op mixes its own value against the skip
    OpReport    -- both of the above for one op of the stack

Shape convention
----------------
Every numeric payload of a report is a tensor whose FIRST axis is the channel
axis and whose REMAINING axes are spatial:

    (K,)        one value per channel, uniform over space   (today's muxes)
    (K, H, W)   one value per channel per cell              (a 2D mux)

Nothing in the rendering or plotting code needs to know which case it is
looking at: text rendering collapses a spatial field to its range, the
plotting helpers hand the field to matplotlib unchanged.  A 2D mux therefore
only has to implement `channel_report` -- the whole reporting path already
works for it.
"""

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "MuxReport",
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


@dataclass
class MuxReport:
    """What one mux is doing.

    ids           channel (token) ids, in channel order
    weights       (K, *spatial) the mux weight of each channel.  For a signed
                  weight mux this is the parameter; for a probabilistic mux it
                  is the (normalized) mixing coefficient.
    probs         (K, *spatial) the probability of each channel, or None when
                  the mux has no probabilistic meaning (a plain signed weight
                  mux has none).  When it exists it is *not* required to be a
                  normalized distribution -- e.g. the ternary mux reports the
                  probability of keeping the continuous value.
    hard_weights  (K, *spatial) the coefficient the hard (eval) forward
                  mixes.  None when the mux makes no difference there.
    mask          (K, *spatial) bool -- where the channel contributes at all.
                  Dead channels are dropped from `ids` by a 1D mux, so this
                  is all-True there; a 2D mux uses it to mark dead cells.
    """

    ids: List[int]
    weights: torch.Tensor
    probs: Optional[torch.Tensor] = None
    hard_weights: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    label: str = ""

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

    def with_weights(self, weights: torch.Tensor) -> "MuxReport":
        """Copy with a different (e.g. hard) set of weights."""
        return replace(self, weights=weights)


@dataclass
class GateReport:
    """How one op mixes its own value against the skip.

    values       (G, *spatial) the branch coefficients.  Branch 0 is the op's
                 own value (the one the "final program" threshold looks at);
                 branch 1 exists for a softmax gate and is the skip.
    hard_values  (G, *spatial) the coefficients the hard (eval) forward uses.
                 For a temperature dependent gate this differs from `values`,
                 which is exactly the train/eval mismatch worth watching.
    """

    values: torch.Tensor
    hard_values: Optional[torch.Tensor] = None
    label: str = ""

    @property
    def is_spatial(self) -> bool:
        return self.values.dim() > 1

    @property
    def keep(self) -> torch.Tensor:
        """Coefficient of the op's own value."""
        return self.values[0]

    @property
    def keep_hard(self) -> torch.Tensor:
        if self.hard_values is None:
            return self.keep
        return self.hard_values[0]


@dataclass
class OpReport:
    """The full report of one op of the stack."""

    index: int
    mux: MuxReport
    gate: GateReport

    @property
    def keep(self) -> float:
        """Scalar summary of the gate's keep coefficient."""
        return scalar_value(self.gate.keep)

    @property
    def keep_hard(self) -> float:
        return scalar_value(self.gate.keep_hard)


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


def format_value(value, fmt: str = "{:.3f}") -> str:
    """Short text for a scalar or for a spatial field (then: its range)."""
    if value is None:
        return ""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    v = value.detach().to(torch.float32).reshape(-1)
    if v.numel() == 0:
        return "-"
    if v.numel() == 1:
        return fmt.format(float(v[0]))
    return "{}..{}".format(fmt.format(float(v.min())), fmt.format(float(v.max())))


def _same(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a.detach(), b.detach()))


def render_mux(mux: MuxReport, name_fn: Callable[[int], str]) -> str:
    """`token(w=+0.010)` / `token(w=-0.010,p=0.120)`, one entry per channel."""
    parts = []
    for j, token_id in enumerate(mux.ids):
        chunk = "{}(w={})".format(name_fn(token_id), format_value(mux.weights[j], "{:+.3f}"))
        if mux.probs is not None and not _same(mux.probs, mux.weights):
            chunk += ",p={}".format(format_value(mux.probs[j]))
        parts.append(chunk)
    return " ".join(parts)


def render_gate(gate: GateReport) -> str:
    """The branch coefficients; `0.731` for a sigmoid gate, `0.30/0.70` for a softmax one."""
    values = gate.values.detach().to(torch.float32)
    values = values.reshape(values.shape[0], -1)
    return "/".join(format_value(v) for v in values)


def render_op(op: OpReport, name_fn: Callable[[int], str]) -> str:
    return render_mux(op.mux, name_fn)


def _quantities(mux: MuxReport, cmap_signed: str, cmap_prob: str):
    """The (label, values, cmap, is_signed) panels one op needs."""
    quantities = [("w", mux.weights, cmap_signed, True)]
    if mux.has_probs and not _same(mux.probs, mux.weights):
        quantities.append(("p", mux.probs, cmap_prob, False))
    return quantities


def _n_panels(mux: MuxReport, quantities) -> int:
    if mux.is_spatial:
        return mux.weights.shape[0] * len(quantities)
    return len(quantities)


def _draw_op_panels(fig, axs, op: OpReport, name_fn, quantities) -> None:
    """Draw one op's panels onto the given axes (left to right)."""
    from matplotlib.colors import TwoSlopeNorm

    mux = op.mux
    col = 0
    for label, values, cmap, signed in quantities:
        v = values.detach().to(torch.float32).cpu()
        if mux.is_spatial:
            vmin = float(v.min())
            vmax = float(v.max())
            if signed:
                lim = max(abs(vmin), abs(vmax))
                norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            else:
                norm = None
            for j, token_id in enumerate(mux.ids):
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
            names = [name_fn(t) for t in mux.ids]
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

    quantities = _quantities(op.mux, cmap_signed, cmap_prob)
    ncols = _n_panels(op.mux, quantities)

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
        quantities = _quantities(op.mux, cmap_signed, cmap_prob)
        per_op.append((op, quantities, _n_panels(op.mux, quantities)))

    nrows = len(per_op)
    ncols = max(n for _, _, n in per_op)

    # a spatial row draws heatmaps and needs to be taller than a 1D bar row
    heights = [3.4 if op.mux.is_spatial else 2.2 for op, _, _ in per_op]

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
