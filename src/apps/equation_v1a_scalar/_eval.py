import torch
import torch.nn as nn
import torch.nn.functional as F

from space.util import eps
from space.token import make_static_token
from space.tokenizer import Vocab

def norm(x, f, rel_eps=1e-2, max_scale=10.0):
    # return f(x)
    vin = torch.std(x, dim=-1, keepdim=True, unbiased=False).detach()
    o = f(x)
    vout = torch.std(o, dim=-1, keepdim=True, unbiased=False).detach()
    scale = vin / (vout + rel_eps * vin + 1e-8)
    scale = torch.clamp(scale, max=max_scale)
    return o * scale

def build_vocab(**_kwargs) -> Vocab:
    """Vocabulary of QG tokens understood by both the embedder and ``qg``."""
    token_specs = [
        # ("_", 1, lambda stack, **kwargs: stack[-1]),
        ("x", 0, lambda stack, **kwargs: kwargs.get("x", 0.0)),
        ("sqrt", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: torch.sqrt(torch.abs(y)))),
        ("cos", 1, lambda stack, **kwargs: norm(stack[-1], torch.cos)),
        ("sin", 1, lambda stack, **kwargs: norm(stack[-1], torch.sin)),
        # ("square", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: y ** 2)),
        # ("cube", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: y ** 3)),
        # ("neg", 1, lambda stack, **kwargs: norm(stack[-1], lambda y: -y)),
        # ("exp", 1, lambda stack, **kwargs: torch.exp(stack[-1])),
        # ("+", 2, lambda stack, **kwargs: stack[-1] + stack[-2]),
        # ("-", 2, lambda stack, **kwargs: stack[-1] - stack[-2]),
        # ("*", 2, lambda stack, **kwargs: stack[-1] * stack[-2]),
        # ("neg", 1, lambda stack, **kwargs: -stack[-1]),
    ]

    tokens = [make_static_token(name, arity, func) for name, arity, func in token_specs]
    return Vocab(tokens=tokens, **_kwargs)

class Evaluator():
    def __init__(self, device='cpu', **_kwargs):
        self.grid = _kwargs.get('grid', [1, 120])
        self.device = device
        assert len(self.grid) == 2

        # self.x = torch.linspace(0, 1, grid_size, device=device)
        # self.y = torch.linspace(0, 1, grid_size, device=device)
        # self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

        self.x = torch.linspace(-8,8,self.grid[-1], device=device)[None,...]


    def __call__(self, closure, **kwargs):
        return closure(x = self.x, **kwargs)

    def metric(self, y_hat, y):
        mse = F.mse_loss(y_hat, y)
        # var = torch.var(y, unbiased=False)
        return mse #/ var

    def plot(self, y_hat_hard, y_hat_soft, y, iter):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.x.cpu()[0], y_hat_soft.detach().cpu()[0], label="y_hat_soft")
        plt.plot(self.x.cpu()[0], y_hat_hard.detach().cpu()[0], label="y_hat_hard")
        plt.plot(self.x.cpu()[0], y.cpu()[0], label="y")
        plt.legend()
        plt.savefig(f"y_hat_{iter}.png")
        plt.close()
        
    # def metric(self, x_hat, x):
    #     mse = F.mse_loss(x_hat, x)
    #     xhf = torch.fft.rfft(x_hat, n=self.x.shape[-1])
    #     xf = torch.fft.rfft(x, n=self.x.shape[-1])
    #     fft = F.mse_loss(xhf.real, xf.real) + F.mse_loss(xhf.imag, xf.imag)
    #     return mse + 0.01*fft

    @property
    def target(self):
        # return torch.sin(torch.sin(0.3 * self.x)) #+ torch.cos(0.005 * self.x)
        # return 3 * self.x
        #### higher frequencies will not work as seen with SIREN
        # return torch.sin(torch.sin(0.3 * self.x)) # works
        # return torch.cos(torch.sin(0.3 * self.x)) # fails (flatline)
        # return torch.cos(torch.sin(0.25 * self.x + 0.5)) - 0.6 # flatline
        # return -torch.cos(torch.sin(0.25 * self.x + 0.5)) - 0.6 # works
        # return -torch.cos(torch.sin(0.3 * self.x)) # flatline
        # return torch.cos(torch.sin(0.3 * (self.x - 2))) # flatline
        return torch.cos(torch.sin(0.3 * self.x)) #+ 0.05 * self.x # sloped line
