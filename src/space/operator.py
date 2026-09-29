import math
import random
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import *
from .token import Token

# class Affine(nn.Module):
#     def __init__(self, scale=1.0, shift=0.0):
#         super().__init__()

#         init = torch.atanh(torch.clamp(torch.tensor(scale) / 10.0, -0.999, 0.999)) * 10
#         self.raw_scale = nn.Parameter(init)
#         init = torch.atanh(torch.clamp(torch.tensor(shift) / 10.0, -0.999, 0.999)) * 10
#         self.raw_shift = nn.Parameter(init)
    
#     def forward(self, _in):
#         return torch.tanh(self.raw_scale / 10.0) * 10.0 * _in + torch.tanh(self.raw_shift / 10.0) * 10.0

class Affine(nn.Module):
    def __init__(self, scale=1.0, shift=0.0, rate=0.1):
        super().__init__()

        self.rate = rate
        self.raw_scale = nn.Parameter(torch.tensor(scale) / self.rate)
        self.raw_shift = nn.Parameter(torch.tensor(shift) / self.rate)
    
    def forward(self, _in, *args):
        return self.raw_scale * self.rate * _in + self.raw_shift * self.rate


class AffineWNorm(nn.Module):
    def __init__(self, scale=1.0, shift=0.0, rate=0.01):
        super().__init__()

        self.rate = rate
        self.raw_scale = nn.Parameter(torch.tensor(scale) / self.rate)
        self.raw_shift = nn.Parameter(torch.tensor(shift) / self.rate)
    
    def forward(self, _in, hardness):
        # _norm = F.layer_norm(_in, (*_in.shape[1:],)) if _in.dim() > 1 else _in
        _norm = _in / (_in.pow(2).mean(dim=tuple(range(1, _in.dim())), keepdim=True) + 1e-8).sqrt() if _in.dim() > 1 else _in
        #### Better, but somehow not the same between hard and soft norm, even if I copy scale. EMA layer norm??
        return self.raw_scale * self.rate * _norm + self.raw_shift * self.rate

@dataclass(eq=False)
class OpArg:
    tokens: List[Token]
    arity: List[int]
    token_id: int = -1
    token_init: Optional[Token] = None
    gate_init: float = 1.0


class OpRep(nn.Module):
    def __init__(self):

        self.n = len(self.tokens)
        ops = [tok() for tok in self.tokens] # default initialization
        if self.token_id >= 0: 
            ops[self.token_id] = self.token_init

        self.ops = nn.ModuleList(ops)

        # implicit representation if needed

    def working_set(self, stack, **kwargs):
        items = [op(stack, **kwargs) for i, op in enumerate(self.ops)]
        target_shape = torch.broadcast_shapes(*[t.shape for t in items])
        items = [t.broadcast_to(target_shape) for t in items]

        working_set = torch.stack(
            items,
            dim=-1,
        ) # CHW, op
        return working_set

class OpParam(nn.Module):
    def __init__(self):

        # self.term_logits = nn.Parameter(0.1 * torch.ones(self.n) / self.n)
        self.term_logits = nn.Parameter(0.1 * torch.rand(self.n))
        if self.token_id >= 0: 
            self.term_logits.data.fill_(0.0)
            self.term_logits.data[self.token_id] = 1.0
            
        # self.preprocess = lambda x, *args: F.layer_norm(x, (x.shape[-1],)) if len(x.shape) > 1 else x
        # self.postprocess = Affine() #WNorm()

        self.preprocess = lambda x, *args: x
        self.postprocess = Affine() 

        # self.norm = lambda x: F.layer_norm(x, (x.shape[-1],)) if len(x.shape) > 1 else x
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(self.gate_init)))
        # self.gate_logit = nn.Parameter((torch.tensor(self.gate_init)))
        # self.gate_logit = nn.Parameter(torch.logit(torch.rand_like(torch.tensor([1.0]))))
        # parameters (logits, affine)

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def gate(self, temp):
        a = torch.sigmoid(self.gate_logit / temp)
        return a, -(a * torch.log(a + eps) + (1-a) * torch.log(1-a + eps))

    def p(self, temp):
        logits = self.term_logits
        logits = logits / temp
        p = F.softmax(logits, dim=0)

        # p = F.gumbel_softmax(logits, dim=0, tau=temp, hard=True)

        entropy = self.term_entropy(p)
        return p, entropy

    def get_token(self, gate_temp=1.0): # go to OpParam but ops to OpRep
        _id = torch.argmax(self.term_logits).item()
        return _id, torch.sigmoid(self.gate_logit / gate_temp)


class OpStack(nn.Module):
    def __init__(self, **kwargs):
        pass        

    def read(self, stack, depth):
        # stack depth, C, H, W
        if depth == 0:
            return stack[1:]
        elif depth == 1:
            return stack[:-1]        
        else:
            fill = torch.ones_like(stack[0])[None,...].expand(abs(depth-1), *stack[0].shape)
            return torch.cat([fill, stack[:-depth]], dim=0)

    def process_stack(self, stack, **kwargs):
        # stack depth, CHW tensor, op
        
        working_set = self.working_set(stack, **kwargs)

        if stack.dim() == 1: # not init
            stack = torch.ones((stack.shape[0], *working_set.shape[:-1]), device=stack.device, dtype=stack.dtype)

        memory = torch.stack(
            [self.read(stack, i) for i in self.arity],
            dim=-1,
        ) # depth, CHW, op

        return stack, working_set, memory

    def stack(self, working_set, memory):
        return torch.cat([memory, working_set[None,...]], dim=0) # depth, BCHW

    def _STE_select(self, p, out, hardness):
        soft_item = torch.sum(p * out, dim=-1)
        hard_item = out[...,torch.argmax(p, dim=0)]
        next_item = soft_item + hardness * (hard_item - soft_item).detach()
        return next_item

    def forward(self, stack, temp=1.0, gate_temp=1.0, hardness=0.0, **kwargs):
        p, term_entropy = self.p(temp)
        a, gate_entropy = self.gate(gate_temp)

        stack = self.preprocess(stack, hardness)

        stack, working_set, memory = self.process_stack(stack, **kwargs) # read and process
        
        memory = self._STE_select(p, memory, hardness) # select from memory
        working_set = self._STE_select(p, working_set, hardness) # select from working set

        working_set = self.postprocess(working_set, hardness) # postprocess with affine transform

        memory = (1 - a)[None,...] * memory + a[None,...] * self.read(stack, 1)
        working_set = (1 - a) * working_set + a * stack[-1] # passthrough


        # memory = memory + self.read(stack, 1)
        # working_set = working_set + stack[-1] # passthrough


        stack = self.stack(working_set, memory) # write back to stack

        return stack, term_entropy, gate_entropy

def Operator(operator_bases):
    class OperatorTemplate(OpArg, *operator_bases, OpStack):
        def __init__(self, init: Dict, **kwargs):
            nn.Module.__init__(self)
            OpArg.__init__(self, **init)
            for cls in operator_bases:
                cls.__init__(self)
            OpStack.__init__(self, **kwargs)

    return OperatorTemplate