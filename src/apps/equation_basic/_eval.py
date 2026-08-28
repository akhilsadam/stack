import torch
import torch.nn as nn
from space.tokenizer import Vocab, Token

def build_vocab(**_kwargs) -> Vocab:
    """Vocabulary of QG tokens understood by both the embedder and ``qg``."""
    token_specs = [
        ("x", 0, lambda stack, **kwargs: kwargs.get("x", 0.0)),
        ("sqrt", 1, lambda stack, **kwargs: torch.sqrt(torch.abs(stack[-1]))),
        ("cos", 1, lambda stack, **kwargs: torch.cos(stack[-1])),
        ("sin", 1, lambda stack, **kwargs: torch.sin(stack[-1])),
        # ("exp", 1, lambda stack, **kwargs: torch.exp(stack[-1])),
        ("square", 1, lambda stack, **kwargs: stack[-1] ** 2),
        ("cube", 1, lambda stack, **kwargs: stack[-1] ** 3),
        ("+", 2, lambda stack, **kwargs: stack[-1] + stack[-2]),
        ("-", 2, lambda stack, **kwargs: stack[-1] - stack[-2]),
        ("*", 2, lambda stack, **kwargs: stack[-1] * stack[-2]),
        ("neg", 1, lambda stack, **kwargs: -stack[-1]),
    ]

    tokens = [Token(name, arity, func) for name, arity, func in token_specs]
    return Vocab(tokens=tokens, **_kwargs)

class Evaluator():
    def __init__(self, device='cpu', **_kwargs):
        # self.grid_size = grid_size
        self.device = device
        self.metric = nn.MSELoss()

        # self.x = torch.linspace(0, 1, grid_size, device=device)
        # self.y = torch.linspace(0, 1, grid_size, device=device)
        # self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="ij")

        self.x = torch.randn(4,4, device=device)

    def __call__(self, closure):
        return closure(x = self.x)

    @property
    def target(self):
        return torch.sin(3 * self.x) + torch.cos(0.005 * self.x)