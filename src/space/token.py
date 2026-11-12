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

class UNK(Token):
    token = "<unk>"
    arity = 1
    def __init__(self):
        super().__init__()

    def forward(self, stack, **kwargs):
        return stack[-1]
