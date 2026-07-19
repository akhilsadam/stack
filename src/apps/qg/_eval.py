from typing import List
import traceback
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import _state, to_physical, to_spectral
from qg.solver.opt.operator.rpn import RPNCompiler

from space.tokens import Token, Vocab


def build_vocab() -> Vocab:
    """Vocabulary of QG tokens understood by both the embedder and ``qg``."""
    token_specs = [
        ("q", 0),
        ("psi", 0),
        ("u", 0),
        ("v", 0),
        ("x", 0),
        ("y", 0),
        ("dx", 1),
        ("dy", 1),
        ("lap", 1),
        ("invlap", 1),
        ("sqrt", 1),
        ("cos", 1),
        ("sin", 1),
        # ("exp", 1),
        ("square", 1),
        ("cube", 1),
        ("+", 2),
        ("-", 2),
        ("*", 2),
        ("jacobian", 2),
        ("neg", 1),
        # ("dealias", 1),
    ]
    tokens = [Token(name, arity) for name, arity in token_specs]
    return Vocab(tokens=tokens)


def make_batch_strings() -> List[str]:
    """RPN expressions that are valid for the ``qg`` compiler."""
    return [
        "q",
        "psi",
        "u v +",
        "q 1.25 *",
        "q psi jacobian",
        "q lap",
        "q psi + sin",
        "x y -",
    ]


class _NoParams:
    """Empty ``pde_params`` — the compiler only reads scalar constants off it."""

    def items(self):
        return []


class QGEvaluator:
    """Callable ``_eval`` backed by the ``qg`` spectral solver.

    Compiles each RPN string and evaluates it on a fixed QG state, returning a
    stacked ``(batch, Ny, Nx)`` tensor of physical fields — the semantic value
    the tokenizer's metric would be computed over.
    """

    def __init__(self, grid_size: int = 32, seed: int = 0, batch=4):
        torch.manual_seed(seed)
        self.grid = CartesianGrid(Nx=grid_size, Ny=grid_size, device=torch.device("cpu"))
        self.derivative = Derivative(self.grid)
        self.compiler = RPNCompiler(self.derivative, _NoParams())

        q_phys = torch.randn(batch, grid_size, grid_size)
        self.state = _state(
            qh=to_spectral(q_phys),
            dt=0.001,
            flow=lambda s: None,
            derivative=self.derivative,
        )

        self.batch = batch
        self.norm = torch.linalg.norm(q_phys[0])

    def eval_one(self, rpn: str) -> torch.Tensor:
        try:
            compiled = self.compiler.compile(rpn)
            field_h = torch.zeros_like(self.state.qh)
            if compiled.linear_operator is not None:
                field_h = field_h + compiled.linear_operator * self.state.qh
            if compiled.nonlinear_source is not None:
                field_h = field_h + compiled.nonlinear_source(self.state)
            val = to_physical(field_h)
            if val.dim() == 4:
                val = val.squeeze()
            return val
        except Exception as e:
            traceback.print_exc()
            print(e)
            return torch.zeros((self.batch, self.grid.Ny, self.grid.Nx), device=self.state.qh.device)

    def __call__(self, strings: List[str]) -> torch.Tensor:
        return torch.stack([self.eval_one(s) for s in strings], dim=0)
