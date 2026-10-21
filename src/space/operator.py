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
from .param import OpParamV2 as OpParam

@dataclass(eq=False)
class OpArg:
    tokens: List[Token]
    arity: List[int]
    token_id: int = -1
    token_init: Optional[Token] = None
    gate_init: float = 1.0
    shape: tuple = (120,)
    max_seq_len: int = 12
    _seq_id: int = 0

    def __post_init__(self):
        self.L = self.max_seq_len
        self.m = len(self.tokens)

class OpRep(nn.Module):
    def __init__(self):

        self.n = len(self.tokens)
        ops = [tok() for tok in self.tokens] # default initialization
        if self.token_id >= 0: 
            ops[self.token_id] = self.token_init

        self.ops = nn.ModuleList(ops)

        # implicit representation if needed

    def working_set(self, stack, **kwargs):
        # Nstack B CHW Op
        items = [op(stack, **kwargs) for i, op in enumerate(self.ops)]
        # print([i.shape for i in items])
        # target_shape = torch.broadcast_shapes(*[t.shape for t in items])
        # items = [t.broadcast_to(target_shape) for t in items]
        # print(stack.shape, [i.shape for i in items])

        working_set = torch.stack(
            items,
            dim=-1,
        ) # B CHW, op
        return working_set


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
        
        # if stack.dim() == 1: # not init
        #     stack = torch.ones((stack.shape[0], *working_set.shape[:-1]), device=stack.device, dtype=stack.dtype)

        working_set = self.working_set(stack, **kwargs)

        memory = torch.stack(
            [self.read(stack, i) for i in self.arity],
            dim=-1,
        ) # depth, CHW, op

        # print(working_set.shape, memory.shape)

        return stack, working_set, memory

    def stack(self, working_set, memory):
        return torch.cat([memory, working_set[None,...]], dim=0) # depth, BCHW

    def _STE_select(self, p, hp, out, hardness):
        soft_item = torch.sum(p * out, dim=-1)
        hard_item = torch.sum(hp * out, dim=-1) # low temp, #out[...,torch.argmax(p, dim=0)]
        next_item = soft_item + hardness * (hard_item - soft_item).detach()
        return next_item

    def _mag(self, ws):
        ws = ws.reshape(-1, ws.shape[-1]) # (BHW*ops, C)
        mag = torch.linalg.norm(ws, dim=0)
        return (mag / (mag.mean() + eps))

    def forward(self, stack, temp=1.0, gate_temp=1.0, hardness=0.0, **kwargs):

        stack = self.preprocess(stack, hardness)
        stack, working_set, memory = self.process_stack(stack, **kwargs) # read and process

        self.cache_mag = self._mag(working_set)
        p, hp, term_entropy = self.p(temp)
        a, gate_entropy = self.gate(gate_temp)

        print(p, self.cache_mag)

        memory = self._STE_select(p, hp, memory, hardness) # select from memory
        working_set = self._STE_select(p, hp, working_set, hardness) # select from working set

        working_set = self.postprocess(working_set, hardness) # postprocess with affine transform

        memory = self.apply_gate(a, memory, self.read(stack, 1))
        working_set = self.apply_gate(a, working_set, stack[-1]) # passthrough gates

        stack = self.stack(working_set, memory) # write back to stack

        return stack, term_entropy, gate_entropy
        
    def get_tokens(self, temp=1.0, gate_temp=1.0):
        return self.channel_ids(cache_mag = self.cache_mag, temp=temp), self.gate_info(gate_temp)

def Operator(operator_bases):
    class OperatorTemplate(OpArg, *operator_bases, OpStack):
        def __init__(self, init: Dict, **kwargs):
            nn.Module.__init__(self)
            OpArg.__init__(self, **init)
            for cls in operator_bases:
                cls.__init__(self)
            OpStack.__init__(self, **kwargs)

    return OperatorTemplate