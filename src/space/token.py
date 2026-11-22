import math
import random
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Token(nn.Module):
    """Token definition (by user)."""
    token = "<base>"
    arity = 1

    def __init__(self):
        super().__init__()

    def forward(self, stack, **kwargs):
        raise NotImplementedError()
    
    def value(self):
        return None

def make_static_token(name, _arity, func):
    class StaticToken(Token):
        token = name.strip()
        arity = _arity
        def __init__(self):
            super().__init__()
            self.forward = func
    return StaticToken



def diverse_gradient(x, grad, eps=1e-8):
    with torch.enable_grad():
        x.requires_grad_(True)
        z = F.normalize(x, dim=-1, eps=eps)
        gram = z @ z.T

        identity = torch.eye(
            x.shape[0],
            device=x.device,
            dtype=x.dtype
        )

        loss = 0.5 * ((gram - identity) ** 2).sum()
        diversity_grad, = torch.autograd.grad(loss, x)

    diversity_grad = diversity_grad * (grad > eps) # disable for non-inputs

    # Remove component parallel to task gradient
    projection = (
        (diversity_grad * grad).sum(dim=-1, keepdim=True)
        / (grad.square().sum(dim=-1, keepdim=True) + eps)
    )

    diversity_grad = diversity_grad - projection * grad

    return grad + 0.1 * diversity_grad

class CVT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x       

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return diverse_gradient(x.view(x.shape[0], -1), grad_output.view(x.shape[0], -1)).view_as(x)


def make_static_diverse_token(name, _arity, func):
    class StaticToken(Token):
        token = name.strip()
        arity = _arity
        def __init__(self):
            super().__init__()
            # args are stack [Nstack B CHW], **kwargs
            self.func = func
        def forward(self, tensor, *args, **kwargs):
            return func(CVT.apply(tensor), *args, **kwargs)     
    return StaticToken

class UNK(Token):
    token = "<unk>"
    arity = 1
    def __init__(self):
        super().__init__()

    def forward(self, stack, **kwargs):
        return stack[-1]
