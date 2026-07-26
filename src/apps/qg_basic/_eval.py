from typing import List, Union, Sequence
from dataclasses import dataclass
import traceback
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import _state, to_physical, to_spectral
from qg.solver.opt.operator.rpn import RPNCompiler
from qg.solver.opt.operator import ImplicitLinearOperator, define_explicit_operator
from qg.solver.integrator import Integrator
from omegaconf import OmegaConf
import logging

from space.tokens import Token, Vocab


@dataclass
class _LaTeXNode:
    text: str
    prec: int
    is_leaf: bool = False


def rpn_to_latex(rpn: Union[str, Sequence[str]]) -> str:
    """Converts an RPN token string or sequence into a formatted LaTeX string."""
    if isinstance(rpn, str):
        tokens = [t for t in rpn.strip().split() if t]
    else:
        tokens = list(rpn)

    if not tokens:
        return ""

    greek_vars = {
        "psi": r"\psi",
        "ph": r"\hat{\psi}",
        "omega": r"\omega",
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "theta": r"\theta",
        "phi": r"\phi",
        "pi": r"\pi",
        "sigma": r"\sigma",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "nu": r"\nu",
        "rho": r"\rho",
        "tau": r"\tau",
        "chi": r"\chi",
        "eta": r"\eta",
        "zeta": r"\zeta",
        "epsilon": r"\epsilon",
    }

    def _paren(node: _LaTeXNode, min_prec: int) -> str:
        if node.prec < min_prec:
            return rf"\left({node.text}\right)"
        return node.text

    stack: List[_LaTeXNode] = []

    for token in tokens:
        t_low = token.lower()

        # Binary operators
        if t_low in ("+", "-", "*", "mul", "/", "div", "^", "pow", "jacobian"):
            if len(stack) < 2:
                raise ValueError(f"Invalid RPN expression: insufficient operands for binary operator '{token}'")
            b = stack.pop()
            a = stack.pop()

            if t_low == "+":
                text = f"{_paren(a, 1)} + {_paren(b, 1)}"
                prec = 1
            elif t_low == "-":
                text = f"{_paren(a, 1)} - {_paren(b, 2)}"
                prec = 1
            elif t_low in ("*", "mul"):
                text = f"{_paren(a, 2)} \cdot {_paren(b, 2)}"
                prec = 2
            elif t_low in ("/", "div"):
                text = rf"\frac{{{a.text}}}{{{b.text}}}"
                prec = 4
            elif t_low in ("^", "pow"):
                text = f"{{{_paren(a, 4)}}}^{{{b.text}}}"
                prec = 3
            elif t_low == "jacobian":
                text = rf"J\left({a.text}, {b.text}\right)"
                prec = 4
            stack.append(_LaTeXNode(text, prec))

        # Unary operators
        elif t_low in (
            "neg", "dx", "dy", "lap", "invlap", "nabla", "del", "laplacian",
            "delta", "del2", "invlaplacian", "lapinv", "sqrt", "cos", "sin",
            "cosh", "sinh", "tanh", "exp", "square", "cube", "abs"
        ):
            if len(stack) < 1:
                raise ValueError(f"Invalid RPN expression: insufficient operands for unary operator '{token}'")
            a = stack.pop()

            if t_low == "neg":
                text = f"-{_paren(a, 2)}"
                prec = 1
            elif t_low in ("dx",):
                text = rf"\partial_x {_paren(a, 4)}"
                prec = 4
            elif t_low in ("dy",):
                text = rf"\partial_y {_paren(a, 4)}"
                prec = 4
            elif t_low in ("lap", "laplacian", "delta", "del2"):
                text = rf"\nabla^2 {_paren(a, 4)}"
                prec = 4
            elif t_low in ("invlap", "invlaplacian", "lapinv"):
                text = rf"\nabla^{{-2}} {_paren(a, 4)}"
                prec = 4
            elif t_low in ("nabla", "del"):
                text = rf"\nabla {_paren(a, 4)}"
                prec = 4
            elif t_low == "sqrt":
                text = rf"\sqrt{{{a.text}}}"
                prec = 4
            elif t_low in ("cos", "sin", "cosh", "sinh", "tanh", "exp"):
                text = rf"\{t_low}\left({a.text}\right)"
                prec = 4
            elif t_low == "square":
                text = f"{{{_paren(a, 4)}}}^2"
                prec = 4
            elif t_low == "cube":
                text = f"{{{_paren(a, 4)}}}^3"
                prec = 4
            elif t_low == "abs":
                text = rf"\left|{a.text}\right|"
                prec = 4
            stack.append(_LaTeXNode(text, prec))

        # Operands (variables / constants)
        else:
            text = greek_vars.get(t_low, token)
            stack.append(_LaTeXNode(text, 4, is_leaf=True))

    if len(stack) != 1:
        raise ValueError(f"Invalid RPN expression: ended with {len(stack)} items on stack for input '{rpn}'")

    return str(stack[0].text)



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

class _NoParams:
    """Empty ``pde_params`` — the compiler only reads scalar constants off it."""

    def items(self):
        return []


# TODO define base class for this
class QGEvaluator:
    """Callable ``_eval`` backed by the ``qg`` spectral solver.

    Compiles each RPN string and evaluates it on a fixed QG state, returning a
    stacked ``(batch, Ny, Nx)`` tensor of physical fields — the semantic value
    the tokenizer's metric would be computed over.
    """

    def __init__(self, grid_size: int = 32, seed: int = 42, batch=2, **kwargs):
        torch.manual_seed(seed)
        self.grid = CartesianGrid(Nx=grid_size, Ny=grid_size, device=torch.device("cpu"))
        self.derivative = Derivative(self.grid)
        self.compiler = RPNCompiler(self.derivative, _NoParams())

        self.batch = batch
        self.steps = kwargs.get('eval_steps', 1)

        self.target_pde = kwargs.get('PDE', None)
        if self.target_pde is None:
            raise ValueError('PDE not found')

        self.param = OmegaConf.create(
            {
            "grid": {
                "Nx": grid_size,
                "Ny": grid_size,
                "Lx": 6.283185307179586,
                "Ly": 6.283185307179586,
                "precision": "float32",
                "device": "cpu"
            },
            "time": {
                "dt": 0.0001,
                "T": 200,
                "save_rate": 1000
            },
            "integrator": {
                "lms": "AB2",
                "ex": "Euler",
                "imex": "CN2",
                "split_bc": False
            },
            "pde": {
                "mu": 0.0,
                "nu": 1.025e-04,
                "B": 0.0,
                "nv": 1.0,
                "penalty": 0.0,
                "friction": None,
                "rossby_radius": None,
                "closure_function": None,
                "closure": 0.0,
                "width": None,
                "rpn": self.target_pde
            },
            "ic": {
                "function": "randn",
                "energy": 0.01,
                "wavenumbers": [10.0, 32.0],
                "seed": seed,
                "n_batch": batch
            },
            "bc": {
                "function": "periodic",
                "inlet_velocity": 0.0,
                "width": 0.05,
                "_min": 0.0,
                "_max": 1.0,
                "sponge": 0.0
            },
            "flow": None,
            "forcing": None,
            "mask": None,
            "fps": 20,
            "profile": False
        })

        from qg.solver.qg import QG
        self.solver = QG(
            self.param
        )

        self.random_state()

    def random_state(self):
        self.q_phys = 4 * torch.rand(self.batch, self.grid.Ny, self.grid.Nx) - 2
        self.norm = torch.linalg.norm(self.q_phys[0])
    
    def construct_state(self):
        return _state(
            qh=to_spectral(self.q_phys),
            dt=self.param.time.dt,
            flow=lambda s: None,
            derivative=self.derivative,
        )
        
        # try:
        #     self.solver.param.pde.rpn = rpn
        #     self.solver.implicit_linear_operator = ImplicitLinearOperator(
        #         self.grid, self.derivative, self.solver.param.pde
        #     )
        #     self.solver.operator = define_explicit_operator(
        #         self.solver.param, self.grid, self.derivative, logging.getLogger("qg"),
        #         args=(self.param.time.dt, self.grid, self.derivative, self.solver.param.pde),
        #         sources=[]
        #     )

        #     self.solver.int = Integrator(self.solver.param.integrator) # reset AB2 histories

        #     state = self.construct_state()
        #     for _ in range(self.steps):
        #         self.solver.step(state)

        #     val = to_physical(state.qh)
        #     if val.dim() == 4:
        #         val = val.squeeze()
                
        #     return val
        # except Exception as e:
        #     traceback.print_exc()
        #     print(e)
        #     return torch.zeros((self.batch, self.grid.Ny, self.grid.Nx), device=self.state.qh.device)

    def eval_one(self, rpn: str) -> torch.Tensor:
        state = self.construct_state()
        try:
            compiled = self.compiler.compile(rpn)
            field_h = torch.zeros_like(state.qh)
            if compiled.linear_operator is not None:
                field_h = field_h + compiled.linear_operator * state.qh
            if compiled.nonlinear_source is not None:
                field_h = field_h + compiled.nonlinear_source(state)
            val = to_physical(field_h)
            if val.dim() == 4:
                val = val.squeeze()
            
            # val is output

            # we want sobolev norm:
            # gxy = torch.gradient(val, dim=(-2, -1))
            # out = torch.stack([val, *gxy], dim=-1)
            out = val

            return out


            # return 1e-3 * val + to_physical(state.qh) # euler
        except Exception as e:
            traceback.print_exc()
            print(e)
            return torch.ones((self.batch, self.grid.Ny, self.grid.Nx), device=state.qh.device) * 1e9


    def target(self) -> torch.Tensor:
        """
        define a simple PDE-target to prove that we can find one.
        """
        return self.eval_one(self.target_pde)

    def estimate(self, rpn: str):
        return self.eval_one(rpn)

    def to_latex(self, rpn: str) -> str:
        return rpn_to_latex(rpn)

    def __call__(self, strings: List[str]) -> torch.Tensor:
        self.random_state()
        return torch.stack([self.eval_one(s) for s in strings], dim=0)

