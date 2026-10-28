import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Sequence
from dataclasses import dataclass
import traceback
import logging

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import _state, to_physical, to_spectral
from qg.solver.opt.operator.rpn import RPNCompiler
from qg.solver.opt.operator import ImplicitLinearOperator, define_explicit_operator
from qg.solver.integrator import Integrator
from omegaconf import OmegaConf

from space.util import eps
from space.token import make_static_token
from space.tokenizer import Vocab

def norm(x, f, rel_eps=1e-2, max_scale=10.0):
    return f(x)

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
        "q": r"q", "psi": r"\psi", "ph": r"\hat{\psi}", "omega": r"\omega", "alpha": r"\alpha", 
        "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta", "theta": r"\theta", 
        "phi": r"\phi", "pi": r"\pi", "sigma": r"\sigma", "lambda": r"\lambda", 
        "mu": r"\mu", "nu": r"\nu", "rho": r"\rho", "tau": r"\tau", "chi": r"\chi", 
        "eta": r"\eta", "zeta": r"\zeta", "epsilon": r"\epsilon",
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
                text = rf"{_paren(a, 2)} \cdot {_paren(b, 2)}"
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

        # Operands
        else:
            text = greek_vars.get(t_low, token)
            stack.append(_LaTeXNode(text, 4, is_leaf=True))

    if len(stack) != 1:
        raise ValueError(f"Invalid RPN expression: ended with {len(stack)} items on stack for input '{rpn}'")

    return str(stack[0].text)


def build_vocab(**_kwargs) -> Vocab:
    """Vocabulary of QG tokens understood by both the embedder and ``qg``."""
    token_specs = [

        
        # ("lω", 0, lambda stack, **kwargs: kwargs.get("lap_func", lambda z: z)(kwargs.get("q", 0.0))),
        ("ω", 0, lambda stack, **kwargs: kwargs.get("q", 0.0)),
        ("ψ", 0, lambda stack, **kwargs: kwargs.get("psi", 0.0)),
        ("u", 0, lambda stack, **kwargs: kwargs.get("u", 0.0)),
        ("v", 0, lambda stack, **kwargs: kwargs.get("v", 0.0)),
        # ("x", 0, lambda stack, **kwargs: kwargs.get("x", 0.0)),
        # ("y", 0, lambda stack, **kwargs: kwargs.get("y", 0.0)),
        ("dx", 1, lambda stack, **kwargs: kwargs.get("dx_func", lambda z: z)(stack[-1])),
        ("dy", 1, lambda stack, **kwargs: kwargs.get("dy_func", lambda z: z)(stack[-1])),
        ("lap", 1, lambda stack, **kwargs: kwargs.get("lap_func", lambda z: z)(stack[-1])),
        # ("invlap", 1, lambda stack, **kwargs: kwargs.get("invlap_func", lambda z: z)(stack[-1])),
        # ("sqrt", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: torch.sqrt(torch.abs(y)))),
        # ("cos", 1, lambda stack, **kwargs: norm(stack[-1], torch.cos)),
        # ("sin", 1, lambda stack, **kwargs: norm(stack[-1], torch.sin)),
        # ("square", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: y ** 2)),
        # ("cube", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: y ** 3)),
        # ("+", 2, lambda stack, **kwargs: stack[-1] + stack[-2]),
        # ("-", 2, lambda stack, **kwargs: stack[-1] - stack[-2]),
        # ("*", 2, lambda stack, **kwargs: stack[-1] * stack[-2]),
        # ("jacobian", 2, lambda stack, **kwargs: kwargs.get("jac_func", lambda a, b: a * b)(stack[-1], stack[-2])),
        # ("neg", 1, lambda stack, **kwargs: -stack[-1]),
    ]

    tokens = [make_static_token(name, arity, func) for name, arity, func in token_specs]
    return Vocab(tokens=tokens, **_kwargs)


class _NoParams:
    """Empty ``pde_params`` — the compiler only reads scalar constants off it."""
    def items(self):
        return []

class Evaluator:
    """Callable ``_eval`` backed by the ``qg`` spectral solver."""

    """A-priori one-step, only eval RHS"""

    def __init__(self, seed: int = 42, device='cpu', **kwargs):
        torch.manual_seed(seed)
        grid_size = kwargs.get('grid', (1, 1, 64, 64))[-1] # ASSUME square

        self.device = device
        self.grid = CartesianGrid(Nx=grid_size, Ny=grid_size, device=torch.device(self.device), precision='float64')
        self.derivative = Derivative(self.grid)
        self.compiler = RPNCompiler(self.derivative, _NoParams())

        self.batch = kwargs.get('grid', (1,1,64,64))[0]
        self.steps = kwargs.get('eval_steps', 1)

        self.base_pde = kwargs.get('base_PDE', '0.0')
        self.target_pde = kwargs.get('PDE', None)
        if self.target_pde is None:
            raise ValueError('PDE not found')

        self.target_pde = self.base_pde + ' ' + self.target_pde + ' +'
        self.dt = 1.0 
        
        print(f"Target PDE: {self.target_pde}")

        self.param = OmegaConf.create({
            "grid": {
                "Nx": grid_size,
                "Ny": grid_size,
                "Lx": 6.283185307179586 * grid_size/64,
                "Ly": 6.283185307179586 * grid_size/64,
                "precision": "float64",
                "device": "cpu"
            },
            "time": {"dt": 0.01, "T": 0.001, "save_rate": 1000},
            "integrator": {"lms": "AB2", "ex": "Euler", "imex": "CN2", "split_bc": False},
            "pde": {
                "mu": 0.0, "nu": 1.025e-04, "B": 0.0, "nv": 1.0,
                "penalty": 0.0, "friction": None, "rossby_radius": None,
                "closure_function": None, "closure": 0.0, "width": None,
                "rpn": self.target_pde
            },
            "ic": {
                "function": "randn", "energy": 0.01,
                "wavenumbers": [10.0, 32.0], "seed": seed, "n_batch": self.batch
            },
            "bc": {
                "function": "periodic", "inlet_velocity": 0.0,
                "width": 0.05, "_min": 0.0, "_max": 1.0, "sponge": 0.0
            },
            "flow": None, "forcing": None, "mask": None, "fps": 20, "profile": False
        })

        from qg.solver.qg import QG
        self.solver = QG(self.param)
        self.random_state()
        
        # --- Precompute fixed grid fields ---
        y, x = torch.meshgrid(
            torch.linspace(0, self.param.grid.Ly, self.grid.Ny, device=self.grid.device),
            torch.linspace(0, self.param.grid.Lx, self.grid.Nx, device=self.grid.device),
            indexing='ij'
        )
        self.x = x[None,None,...].expand(self.batch, -1, -1, -1)
        self.y = y[None,None,...].expand(self.batch, -1, -1, -1)

        # --- Precompute fixed state fields ---
        state = self.construct_state()
        self.q = self.q_phys.detach().clone()
        self.psi = to_physical(self.derivative.inv_laplacian * (state.qh))
        self.u = to_physical(-self.derivative.dy * (self.derivative.inv_laplacian * (state.qh)))
        self.v = to_physical(self.derivative.dx * (self.derivative.inv_laplacian * (state.qh)))

        # --- Precompute operators ---
        dx_scale = self.param.grid.Lx / self.param.grid.Nx
        dy_scale = self.param.grid.Ly / self.param.grid.Ny
        d2_scale = dx_scale**2 + dy_scale**2
        
        print(d2_scale)

        self.dx_func = lambda z: to_physical(self.derivative.dx * (to_spectral(z)))
        self.dy_func = lambda z: to_physical(self.derivative.dy * (to_spectral(z)))
        self.lap_func = lambda z: to_physical(self.derivative.laplacian * (to_spectral(z)))
        self.invlap_func = lambda z: to_physical(self.derivative.inv_laplacian * (to_spectral(z)))
        # self.jac_func = lambda a, b: to_physical(self.derivative.jacobian(to_spectral(a), to_spectral(b)))

        # --- precompute a-priori step with Euler
        self._base_step = self.eval_one(self.base_pde)
        self._target = self.q + self.dt * self.eval_one(self.target_pde)
        self._base = self.q + self.dt * self._base_step

    def random_state(self):
        self.q_phys = 4 * torch.rand(self.batch, self.grid.Ny, self.grid.Nx) - 2
        self.q_phys = self.solver.nn_step(self.q_phys, n_steps=10000, dt=0.01)[:,0]
        self.norm = torch.linalg.norm(self.q_phys[0])
        
    def construct_state(self):
        return _state(
            qh=self.derivative.dealias(to_spectral(self.q_phys)),
            dt=self.param.time.dt,
            derivative=self.derivative,
        )

    def eval_one(self, rpn: str) -> torch.Tensor:
        state = self.construct_state()
        try:
            compiled = self.compiler.compile(rpn)
            field_h = torch.zeros_like(state.qh)
            if compiled.linear_operator is not None:
                field_h = field_h + (compiled.linear_operator * state.qh)
            if compiled.nonlinear_source is not None:
                field_h = field_h + compiled.nonlinear_source(state)

            # field_h[...,0,0] = 0.0 # toss zeroth mode

            val = to_physical(self.derivative.dealias(field_h))
            # if val.dim() >= 4:
            #     val = val.squeeze()
            
            return val
        except Exception as e:
            raise ValueError("Check PDE input...")
            # traceback.print_exc()
            # print(e)
            # return torch.zeros((self.batch, 1, self.grid.Ny, self.grid.Nx), device=state.qh.device)

    @property
    def target(self) -> torch.Tensor:
        return self._target

    def __call__(self, closure, **kwargs):
        # Evaluate closure with precomputed state and operators
        dqdt = self._base_step + closure(q=self.q, psi=self.psi, u=self.u, v=self.v, 
                       x=self.x, y=self.y, 
                       dx_func=self.dx_func, dy_func=self.dy_func,
                       lap_func=self.lap_func, invlap_func=self.invlap_func, **kwargs)

        return self.q + self.dt * dqdt # euler step

    def metric(self, y_hat, y):
        # print(y_hat.shape, y.shape)
        # y_hatn = y_hat / torch.mean(y_hat.abs())
        # yn = y / torch.mean(y.abs())
        # mse = F.mse_loss(y_hatn, yn)
        #/ (torch.var(y, unbiased=False) + 1e-14)


        mse = F.mse_loss(y_hat, y) 
        return mse

    def plot(self, y_hat_hard, y_hat_soft, y, iter):
        from jpcm import draw
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        
        im0 = axs[0,0].imshow(self._base.detach().cpu()[0,0], cmap=draw.cmap)
        axs[0,0].set_title("Base ω (ULES)")
        fig.colorbar(im0, ax=axs[0,0])

        im1 = axs[0,1].imshow(y_hat_soft.detach().cpu()[0,0], cmap=draw.cmap)
        axs[0,1].set_title("soft closed ω (LES)")
        fig.colorbar(im1, ax=axs[0,1])
        
        im2 = axs[0,2].imshow(y_hat_hard.detach().cpu()[0,0], cmap=draw.cmap)
        axs[0,2].set_title("hard closed ω (LES)")
        fig.colorbar(im2, ax=axs[0,2])
        
        im3 = axs[0,3].imshow(y.detach().cpu()[0,0], cmap=draw.cmap)
        axs[0,3].set_title("DNS ω")
        fig.colorbar(im3, ax=axs[0,3])
        
        # im0 = axs[1,0].imshow(self.q.detach().cpu()[0,0], cmap=draw.cmap)
        # axs[1,0].set_title("vorticity (state)")
        # fig.colorbar(im0, ax=axs[1,0])
        
        im4 = axs[1,0].imshow((y - self._base).cpu()[0,0], cmap=draw.cmap)
        axs[1,0].set_title("y - base")
        fig.colorbar(im4, ax=axs[1,0])

        im5 = axs[1,1].imshow((y_hat_soft - self._base).detach().cpu()[0,0], cmap=draw.cmap)
        axs[1,1].set_title("soft closure only (signal)")
        fig.colorbar(im5, ax=axs[1,1])
        
        im6 = axs[1,2].imshow((y_hat_hard - self._base).detach().cpu()[0,0], cmap=draw.cmap)
        axs[1,2].set_title("hard closure only (signal)")
        fig.colorbar(im6, ax=axs[1,2])

        im7 = axs[1,3].imshow((y_hat_soft - y).detach().cpu()[0,0], cmap=draw.cmap)
        axs[1,3].set_title("soft closure err")
        fig.colorbar(im7, ax=axs[1,3])

        plt.suptitle(f'a-priori closure at it={iter} for {self.target_pde}')
        plt.tight_layout()
        plt.savefig(f"y_hat_{iter}.png")
        plt.close()