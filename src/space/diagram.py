"""Terminal diagram of the stack."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
import torch

__all__ = ["Diagram", "render_diagram", "OP_COLOR", "TERM_COLOR"]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

OP_COLOR = "\033[38;5;39m"
TERM_COLOR = "\033[38;5;41m"
FG = "\033[38;5;255m"


@dataclass
class OpNode:
    index: int
    token: str
    arity: int
    weight: float
    skip: float
    # (name, mean, std) per channel -- std is 0 for a 1D mux, and the spread
    # of the field for a spatial one
    channels: List[Tuple[str, float, float]] = field(default_factory=list)


@dataclass
class Diagram:
    nodes: List[OpNode]
    capacity: int

    @staticmethod
    def from_reports(
        reports,
        arities: Sequence[int],
        name_fn: Callable[[int], str],
        capacity: int,
    ) -> "Diagram":
        nodes = []
        for r in reports:
            op = r.final  # every report of the op, combined
            if op.size == 0:
                nodes.append(OpNode(index=r.index, token=".", arity=0, weight=0.0, skip=r.skip))
                continue

            # a 1D mux has one weight per channel; a spatial one has a whole
            # field, which the terminal summarises as mean +- std
            w = op.weights.detach().to(torch.float32)
            w = w.reshape(w.shape[0], -1)  # (K, cells)
            means = w.mean(dim=1).cpu().tolist()
            stds = w.std(dim=1, unbiased=False).cpu().tolist()

            channels = [
                (name_fn(int(tok)), mean, std)
                for tok, mean, std in zip(op.ids, means, stds)
            ]
            best = max(range(len(means)), key=lambda i: abs(means[i]))
            tok = op.ids[best]

            # two rows that are not tokens, added after the best channel is
            # picked so they can never be mistaken for one of the op's terms:
            #   "1"     the constant the op adds on top of the weighted mix
            #           (the affine's shift; 0 when there is none)
            #   "skip"  how much of the previous value the gate passes through
            offset = 0.0 if op.offset is None else float(torch.as_tensor(op.offset).float().mean())
            channels.append(("1", offset, 0.0))
            # channels.append(("skip", r.skip, 0.0))

            nodes.append(
                OpNode(
                    index=r.index,
                    token=name_fn(int(tok)),
                    arity=arities[tok] if tok < len(arities) else 1,
                    weight=abs(means[best]),
                    skip=r.skip,
                    channels=channels,
                )
            )
        return Diagram(nodes=nodes, capacity=capacity)

    def rpn(self) -> List[str]:
        return [n.token for n in self.nodes if n.token != "."]

    def reverse_equation(self) -> str:
        return " ".join(self.rpn())


def render_diagram(diag: Diagram, width: int = 14, threshold: float = 0.05, color: bool = True) -> str:
    # Gather all unique variable/token names across all steps in order of appearance
    var_names = []
    seen = set()
    for node in diag.nodes:
        if node.token == ".":
            continue
        if node.token not in seen:
            seen.add(node.token)
            var_names.append(node.token)
        for tok, _, _ in node.channels:
            if tok not in seen:
                seen.add(tok)
                var_names.append(tok)

    n_steps = len([n for n in diag.nodes if n.token != "."])

    # Map each (step_idx, var_name) -> (mean, std)
    grid = {}
    step_col = 0
    for node in diag.nodes:
        if node.token == ".":
            continue
        for tok, mean, std in node.channels:
            grid[(tok, step_col)] = (mean, std)
        step_col += 1

    # a spatial channel is summarised as mean +- std; a 1D one is just a value
    def _cell(mean, std):
        if std > 0:
            return f"{mean:+.3f}±{std:.2f}"
        return f"{mean:+.3f}"

    rows = []
    for var in var_names:
        cells = []
        for s in range(n_steps):
            v = grid.get((var, s))
            cells.append("" if v is None else _cell(v[0], v[1]))
        rows.append((var, cells))

    col_w = max([len(c) for _, cells in rows for c in cells] + [len("var")]) + 2

    lines = [
        BOLD + FG + "stack program" + RESET,
        f"   {'var':<6} " + "".join(f"{s:<{col_w}}" for s in range(n_steps)),
    ]

    for var, cells in rows:
        lines.append(f"   {var:<6} " + "".join(f"{c:<{col_w}}" for c in cells))

    lines.append("")
    lines.append(BOLD + FG + "equation (RPN)" + RESET)
    eq_str = diag.reverse_equation()
    lines.append(f"   {eq_str}")

    text = "\n".join(lines)
    if color:
        return text

    out, i = [], 0
    while i < len(text):
        if text[i] == "\033":
            j = text.find("m", i)
            i = len(text) if j < 0 else j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)