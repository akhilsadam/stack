import torch
import torch.nn as nn
import torch.nn.functional as F

from space.tokenizer import Vocab, make_static_token

def build_vocab(**_kwargs) -> Vocab:
    """Vocabulary of QG tokens understood by both the embedder and ``qg``."""
    token_specs = [
        ("x", 0, lambda stack, **kwargs: kwargs.get("x", 0.0)),
        ("sqrt", 1, lambda stack, **kwargs: torch.sqrt(torch.abs(stack[-1]))),
        ("cos", 1, lambda stack, **kwargs: torch.cos(stack[-1])),
        ("sin", 1, lambda stack, **kwargs: torch.sin(stack[-1])),
        # ("exp", 1, lambda stack, **kwargs: torch.exp(stack[-1])),
        # ("square", 1, lambda stack, **kwargs: stack[-1] ** 2),
        # ("cube", 1, lambda stack, **kwargs: stack[-1] ** 3),
        # ("+", 2, lambda stack, **kwargs: stack[-1] + stack[-2]),
        # ("-", 2, lambda stack, **kwargs: stack[-1] - stack[-2]),
        # ("*", 2, lambda stack, **kwargs: stack[-1] * stack[-2]),
        ("neg", 1, lambda stack, **kwargs: -stack[-1]),
    ]

    # use helper to make a blank class instances that are initialized later
    tokens = [make_static_token(name, arity, func) for name, arity, func in token_specs]
    return Vocab(tokens=tokens, **_kwargs)

class Evaluator():
    def __init__(self, device='cpu', **_kwargs):
        # self.grid_size = grid_size
        self.device = device

        # self.x = torch.linspace(0, 1, grid_size, device=device)
        # self.y = torch.linspace(0, 1, grid_size, device=device)
        # self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

        self.x = torch.linspace(-4,4,120, device=device)

    def __call__(self, closure, **kwargs):
        return closure(x = self.x, **kwargs)

    def metric(self, y_hat, y):
        mse = F.mse_loss(y_hat, y)
        # var = torch.var(y, unbiased=False)
        return mse #/ var

    def plot(self, y_hat_hard, y_hat_soft, y, iter):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.x.cpu(), y_hat_soft.detach().cpu(), label="y_hat_soft")
        plt.plot(self.x.cpu(), y_hat_hard.detach().cpu(), label="y_hat_hard")
        plt.plot(self.x.cpu(), y.cpu(), label="y")
        plt.legend()
        plt.savefig(f"y_hat_{iter}.png")
        plt.close()
        
    # def metric(self, x_hat, x):
    #     mse = F.mse_loss(x_hat, x)
    #     xhf = torch.fft.rfft(x_hat, n=self.x.shape[-1])
    #     xf = torch.fft.rfft(x, n=self.x.shape[-1])
    #     fft = F.mse_loss(xhf.real, xf.real) + F.mse_loss(xhf.imag, xf.imag)
    #     return mse + 0.0001*fft

    @property
    def target(self):
        # return torch.sin(3 * self.x) #+ torch.cos(0.005 * self.x)
        # return 3 * self.x
        # return torch.sin(self.x)
        return torch.sin(torch.sin(self.x))