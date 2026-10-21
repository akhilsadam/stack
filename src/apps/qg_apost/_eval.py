import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Sequence
from dataclasses import dataclass
import traceback
import logging

from qg.solver.opt.basis import to_physical, to_spectral
from qg.solver.qg import QG
from qg.solver.integrator import Integrator
from omegaconf import OmegaConf

from space.util import eps
from space.token import make_static_token
from space.tokenizer import Vocab


# --- monkey patch torch.nan_to_num to support complex autograd tensors ---
_orig_nan_to_num = torch.nan_to_num
def _safe_nan_to_num(input, *args, **kwargs):
    if input.is_complex() and input.requires_grad:
        return torch.complex(
            _orig_nan_to_num(input.real, *args, **kwargs),
            _orig_nan_to_num(input.imag, *args, **kwargs)
        )
    return _orig_nan_to_num(input, *args, **kwargs)
torch.nan_to_num = _safe_nan_to_num
# ------------------------------------------------------------------------

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
        # ("u", 0, lambda stack, **kwargs: kwargs.get("u", 0.0)),
        # ("v", 0, lambda stack, **kwargs: kwargs.get("v", 0.0)),
        # ("x", 0, lambda stack, **kwargs: kwargs.get("x", 0.0)),
        # ("y", 0, lambda stack, **kwargs: kwargs.get("y", 0.0)),
        # ("dx", 1, lambda stack, **kwargs: kwargs.get("dx_func", lambda z: z)(stack[-1])),
        # ("dy", 1, lambda stack, **kwargs: kwargs.get("dy_func", lambda z: z)(stack[-1])),
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


class Evaluator:
    """Callable ``_eval`` backed by the ``qg`` spectral solver."""

    def __init__(self, seed: int = 42, device='cpu', **kwargs):
        torch.manual_seed(seed)
        grid_size = kwargs.get('grid', (1, 1, 64, 64))[-1] # ASSUMES square

        self.device = device

        self.batch = kwargs.get('grid', (1,1,64,64))[0]
        self.steps = kwargs.get('eval_steps', 1)

        self.base_pde = kwargs.get('base_PDE', '0.0')
        self.target_pde = kwargs.get('PDE', None)
        if self.target_pde is None:
            raise ValueError('PDE not found')

        self.target_pde = self.base_pde + ' ' + self.target_pde + ' +'
        
        # Use the configuration time step
        self.dt = kwargs.get('dt', 1.0) 
        
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
            "time": {"dt": self.dt, "T": 0.001, "save_rate": 1000},
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
        self.base_param = self.param.copy()
        self.base_param.pde.rpn = self.base_pde

        self.solver = QG(self.param)
        self.random_state()
        
        # --- Precompute fixed grid fields ---
        y, x = torch.meshgrid(
            torch.linspace(0, self.param.grid.Ly, self.param.grid.Ny, device=self.param.grid.device),
            torch.linspace(0, self.param.grid.Lx, self.param.grid.Nx, device=self.param.grid.device),
            indexing='ij'
        )
        self.x = x[None,None,...].expand(self.batch, -1, -1, -1)
        self.y = y[None,None,...].expand(self.batch, -1, -1, -1)

        # --- Initial state vorticity ---
        self.q = self.q_phys.detach().clone()

        # --- Precompute operators ---
        dx_scale = self.param.grid.Lx / self.param.grid.Nx
        dy_scale = self.param.grid.Ly / self.param.grid.Ny
        d2_scale = dx_scale**2 + dy_scale**2

        self.dx_func = lambda z: to_physical(self.closed_solver.derivative.dx * (to_spectral(z)))
        self.dy_func = lambda z: to_physical(self.closed_solver.derivative.dy * (to_spectral(z)))
        self.lap_func = lambda z: to_physical(self.closed_solver.derivative.laplacian * (to_spectral(z)))
        self.invlap_func = lambda z: to_physical(self.closed_solver.derivative.inv_laplacian * (to_spectral(z)))

    def random_state(self):
        """generate random vorticity data"""
        q_phys = 4 * torch.rand(self.batch, self.param.grid.Ny, self.param.grid.Nx) - 2
        self.q_phys = self.solver.nn_step(q_phys, n_steps=10000, dt=0.01)[:,:,0,...] # BTCHW
        self.norm = torch.linalg.norm(self.q_phys[0])
        
        # get cloned caches & states
        self.base_solver = QG(self.base_param)
        self.closed_solver = QG(self.base_param)
        # we don't clone integrator caches, so operator will have some small-dt spinup error

        # rollout true and unclosed trajectories
        self._target = self.solver.nn_step(self.q_phys[:, -1], n_steps=self.steps, dt=self.dt)[:, -1]
        self._base = self.base_solver.nn_step(self.q_phys[:, -1], n_steps=self.steps, dt=self.dt)[:, -1]
        self.cache_explicit_opt = self.base_solver.operator
        
    def get_fields(self, state):
        q = to_physical(state.qh)[:,None,...]
        psi = to_physical(state.ph)[:,None,...]
        u = to_physical(state.uh)[:,None,...]
        v = to_physical(state.vh)[:,None,...]
        return {'q': q, 'psi': psi, 'u': u, 'v': v}

    def update_operator(self, solver, closure, kwargs):
        self.closed_solver.int = Integrator(self.param.integrator) # reset integrator

        def wrap(state):
            f = self.cache_explicit_opt.source(state)
            pi = closure(
                **self.get_fields(state),
                x=self.x, y=self.y,
                dx_func=self.dx_func, dy_func=self.dy_func,
                lap_func=self.lap_func, invlap_func=self.invlap_func, **kwargs
            )[:,0,:,:] * self.dt # BHW
            pi_h = self.closed_solver.derivative.dealias(to_spectral(pi))
            return f + pi_h

        solver.operator.source = wrap

    @property
    def target(self) -> torch.Tensor:
        return self._target

    def __call__(self, closure, **kwargs):
        self.update_operator(self.closed_solver, closure, kwargs)
        
        # BPTT
        q_closed = self.closed_solver.nn_step(self.q_phys[:, -1].detach().clone(), n_steps=self.steps, dt=self.dt)[:, -1]
        #### loss 5.46e-6

        # # Schrodinger Bridge
        # with torch.no_grad():
        #     q_minus = self.closed_solver.nn_step(self.q_phys[:, -1].detach().clone(), n_steps=self.steps-1, dt=self.dt)[:, -1,...]
        # q_minus.requires_grad_(True)
        # q_closed = self.closed_solver.nn_step(q_minus[:,0], n_steps=1, dt=self.dt)[:, -1]
        # #### loss 9.28e-6

        # # Schrodinger Bridge; 2 steps
        # with torch.no_grad():
        #     q_minus = self.closed_solver.nn_step(self.q_phys[:, -1].detach().clone(), n_steps=self.steps-2, dt=self.dt)[:, -1,...]
        # q_minus.requires_grad_(True)
        # q_closed = self.closed_solver.nn_step(q_minus[:,0], n_steps=2, dt=self.dt)[:, -1]
        # #### loss 8.4e-6
        
        # # Schrodinger Bridge; n-1 steps
        # with torch.no_grad():
        #     q_minus = self.closed_solver.nn_step(self.q_phys[:, -1].detach().clone(), n_steps=1, dt=self.dt)[:, -1,...]
        # q_minus.requires_grad_(True)
        # q_closed = self.closed_solver.nn_step(q_minus[:,0], n_steps=self.steps-1, dt=self.dt)[:, -1]
        # #### loss 5.46e-6

        # # Schrodinger Bridge; random steps
        # nt = torch.randint(low=1, high=self.steps-1, size=(1,)).item()
        # with torch.no_grad():
        #     q_minus = self.closed_solver.nn_step(self.q_phys[:, -1].detach().clone(), n_steps=nt, dt=self.dt)[:, -1,...]
        # q_minus.requires_grad_(True)
        # q_closed = self.closed_solver.nn_step(q_minus[:,0], n_steps=self.steps-nt, dt=self.dt)[:, -1]
        # #### loss 6.89e-6

        return q_closed

    def metric(self, y_hat, y):
        return F.mse_loss(y_hat, y) 

    def plot(self, y_hat_hard, y_hat_soft, y, iter):
        from jpcm import draw
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        
        im0 = axs[0,0].imshow(self._base.detach().cpu()[0,0], cmap=draw.cmap)
        axs[0,0].set_title(f"Base ω ({self.steps}-step ULES)")
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

        plt.suptitle(f'a-posteriori multi-step closure at it={iter} for {self.target_pde}')
        plt.tight_layout()
        plt.savefig(f"y_hat_{iter}.png")
        plt.close()