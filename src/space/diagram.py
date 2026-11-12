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
    keep: float
    channels: List[Tuple[str, float]] = field(default_factory=list)


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
            mux = r.mux
            if mux is None or getattr(mux, "size", 0) == 0 or mux.ids is None:
                nodes.append(OpNode(index=r.index, token=".", arity=0, weight=0.0, keep=r.keep))
                continue

            ids = mux.ids.detach().cpu().tolist() if hasattr(mux.ids, "detach") else list(mux.ids)
            weights = (
                mux.weights.detach().to(torch.float32).reshape(-1).cpu().tolist()
                if hasattr(mux.weights, "detach")
                else list(mux.weights)
            )

            channels = [(name_fn(int(tok)), float(wt)) for tok, wt in zip(ids, weights)]
            best = int(torch.argmax(torch.tensor([abs(w) for w in weights]))) if weights else 0
            tok = ids[best] if ids else 0
            best_wt = weights[best] if weights else 0.0

            nodes.append(
                OpNode(
                    index=r.index,
                    token=name_fn(int(tok)),
                    arity=arities[tok] if tok < len(arities) else 1,
                    weight=float(abs(best_wt)),
                    keep=r.keep,
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
        for tok, _ in node.channels:
            if tok not in seen:
                seen.add(tok)
                var_names.append(tok)

    n_steps = len([n for n in diag.nodes if n.token != "."])
    
    # Map each (step_idx, var_name) -> weight
    grid = {}
    step_col = 0
    for node in diag.nodes:
        if node.token == ".":
            continue
        for tok, w in node.channels:
            grid[(tok, step_col)] = w
        step_col += 1

    lines = [
        BOLD + FG + "stack program" + RESET,
        f"   {'var':<6} " + "".join(f"{s:<10}" for s in range(n_steps))
    ]

    for var in var_names:
        row_vals = []
        for s in range(n_steps):
            w = grid.get((var, s))
            row_vals.append(f"{w:+.3f}" if w is not None else "")
        lines.append(f"   {var:<6} " + "".join(f"{val:<10}" for val in row_vals))

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
