import math
import random
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# TODO extend to multidimensionality / matrix / vector / tensor equations
## needs to add physical dimension to token
## needs a PhysicalTensor class to extract tensor constants
## needs to handle tensor contractions and maybe Einstein summation notation...
## quite difficult for now as solver doesn't support either yet

from .util import *
from .token import *
from .operator import *
from .report import render_op, render_gate, plot_stack
from .diagram import Diagram, render_diagram

class Vocab:

    def __init__(
        self,
        tokens: List[Token],
        seq_len: int = 64,
        operator_bases = [OpRep, OpParam],
        **kwargs,
    ):
        self.seq_len = seq_len
        # self.tokens = [UNK, *tokens]
        self.tokens = tokens
        self.arity = [t.arity for t in self.tokens]

        self.token_to_id = {
            token.token: i for i, token in enumerate(self.tokens)
        }
        self.id_to_token = {
            i: token.token for i, token in enumerate(self.tokens)
        }
        self.id_to_arity = {
            i: token.arity for i, token in enumerate(self.tokens)
        }

        print([token.token for token in self.tokens])
        # self.pad_token_id = self.token_to_id["<unk>"]
        self._Operator = Operator(operator_bases)
        self.shape = kwargs['grid'] # B [CHW]

    def __len__(self):
        return len(self.tokens)

    def rep_from_str(self, token_str: str) -> int:
        _id = self.token_to_id.get(token_str.strip().lower(), -1)
        return _id 

    def tokenize_str(self, _str: str):
        ops = []
        toks = _str.split(" ")

        # passthrough = 1.0
        passthrough = 0.5
        
        init_kwargs = {
            "tokens": self.tokens,
            "arity": self.arity,
            "token_id": -1,
            "token_init": None,
            "gate_init": passthrough,
            "shape": self.shape,
            "max_seq_len": self.seq_len,
            "_seq_id": 0,
        }

        for i, s in enumerate(toks):
            _id = self.rep_from_str(s)
            if _id == -1:
                t = None
                passthrough = passthrough  #torch.rand((1,)).item()
            else:
                passthrough = passthrough 
                # passthrough = 0.1 # regardless we use passthrough to start from nothing
                t = self.tokens[_id]()
            ops.append(self._Operator(
                init = {
                    **init_kwargs,
                    "token_id": _id,
                    "token_init": t,
                    "gate_init": passthrough,
                    "_seq_id": i,
                },
            ))

        if len(toks) < self.seq_len:
            ops.extend([self._Operator(init={
                **init_kwargs, 
                "_seq_id": i, 
            }) for i in range(len(toks), self.seq_len)])

        # print(ops[0].__dict__)
        return ops

class Stack(nn.Module):
    def __init__(self, init_str: str, vocab: Vocab, T0=20, **kwargs):
        super().__init__()
        self.vocab = vocab

        self.ops = nn.ModuleList(
            self.vocab.tokenize_str(init_str)
        )

        self.capacity = kwargs.get("capacity", 4)
        # self.stack = self.init_stack
        self.register_buffer('stack', torch.ones((self.capacity, *self.vocab.shape)))

        self.T0 = T0
        self.temp = nn.Parameter(torch.ones(self.vocab.seq_len))
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

        self.output_logits = nn.Parameter(torch.zeros(min(self.vocab.seq_len, self.capacity)))
        self.output_logits.data[-1] = 1.0
        self.output_temp = 1.0

        self._iter = kwargs.get('_iter', 2000)

    def T(self, temp, scale=1.0):
        return scale * F.sigmoid(temp) + eps # 0.01

    # def init_stack(self, device=None, dtype=torch.float32):
    #     return torch.ones((self.capacity, *self.vocab.shape), dtype=dtype, device=device)
    #     # return torch.stack([torch.tensor(self.vocab.tokens[0]()(None),device=device) for i in range(self.capacity)], dim=0)

    def reset_accum(self):
        self._term_entropy = torch.tensor(0.0)

    def forward(self, stack_in=None, _eval=False, **kwargs):
        stack = stack_in or self.stack
        temp = self.T(self.temp, self.T0)
        gate_temp = self.T(self.gate_temp)

        if _eval:
            gate_temp = self.T(self.gate_temp, 0.0)
            kwargs['hardness'] = 1.0

        _entropy = 0.0
        for i, op in enumerate(self.ops):
            stack, _term_entropy, _gate_entropy = op(stack, temp = temp[i], gate_temp = gate_temp, **kwargs)
            # stack_in = stack_in[-self.max_depth:]
            _entropy = _entropy + _term_entropy + _gate_entropy / len(self.vocab.tokens)

        # elems = torch.stack(stack_in[-self.vocab.seq_len:], dim=-1)
        # attn_weights = F.softmax(self.output_logits / self.output_temp, dim=0)
        # out = torch.sum(elems * attn_weights, dim=-1) # provides gradient, bypass constants that gradient block
        # return out
        
        # assign cached vars
        self._term_entropy = self._term_entropy + _entropy

        return stack[-1] # last element

    @torch.no_grad()
    def report(self, threshold = 0.05):
        """Structured report of the stack: mux weights/probabilities, gates.

        The list is parallel to ``self.ops``; each entry carries the mux of
        that op (which channels are on, with what weight, with what
        probability) and its gate.  A spatial mux plugs into this without any
        change here -- see ``space/report.py``.
        """
        temp = self.T(self.temp, self.T0)
        gate_temp = self.T(self.gate_temp)

        reports = []
        for i, op in enumerate(self.ops):
            r = op.report(temp=temp[i], gate_temp=gate_temp)
            r.index = i
            reports.append(r)
        return reports

    @torch.no_grad()
    def diagram(self, threshold = 0.05):
        """The stack, drawn as a terminal diagram.

        The horizontal axis is the operand stack (top of the stack on the
        left, as the machine reads it), the vertical axis is the ops in
        execution order.  See ``space/diagram.py``.
        """
        return Diagram.from_reports(
            self.report(threshold=threshold),
            self.vocab.arity,
            self.vocab.id_to_token.get,
            self.capacity,
        )



    @torch.no_grad()
    def detokenize(self, threshold=0.05, color=True) -> str:
        reports = self.report(threshold=threshold)
        name = self.vocab.id_to_token.get

        diag = Diagram.from_reports(reports, self.vocab.arity, name, self.capacity)
        picture = render_diagram(diag, threshold=threshold, color=color)

        return f"\n\n{picture}\n\n"

    @torch.no_grad()
    def plot_report(self, path = None, threshold = 0.05):
        """Plot the mux weights (and probabilities) of the whole stack."""
        return plot_stack(self.report(threshold=threshold), self.vocab.id_to_token.get, path=path)

            
    
    def _train(self, _eval, lr=1e-2, **kwargs):
        opt = create_optimizer(self, torch.optim.Adam, lr)
        for i in range(self._iter):
            opt.zero_grad()

            self.reset_accum()
            z_hat = _eval(self.forward)

            # log schedule
            # T_schedule = math.exp(-i/200)
            # loss_temp = F.relu(self.T(self.temp) - T_schedule).pow(2.0).mean()
            # loss_gate = F.relu(self.T(self.gate_temp) - T_schedule).pow(2.0).mean()

            loss_target = _eval.metric(z_hat, _eval.target) # full-batch

            # random subset
            # mask = torch.randperm(z_hat.shape[0], device=z_hat.device)[:10]
            # loss_target = _eval.metric(z_hat[mask], _eval.target[mask])

            loss_entropy = self._term_entropy
            loss_temp = self.T(self.temp).pow(2.0).mean()
            loss_gate = self.T(self.gate_temp).pow(2.0).mean()

            logger.debug(f"Loss target: {loss_target.item():.2e}, loss_entropy: {loss_entropy.item():.2e}, temp: {F.sigmoid(self.temp).mean().item():.2e}, loss gate: {loss_gate.item():.2e}")


            # # loss = loss_target + 1e-4 * loss_temp 
            # # loss = max(loss_target, loss_temp / 20)
            # a = (loss_temp / 20) > loss_target
            # b = 0.05 if a else 1e-4 
            # c = 0 #1e-4
            # loss = loss_target + b * loss_temp + c * loss_gate
            # loss = max(loss_target, loss_entropy + 1e-1)
            

            # if (loss_target > 1e-3 and loss_entropy > 1e-2):
            #     # not converged yet; don't optimize gate
            #     self.gate_temp.requires_grad_(False)
            # else:
            #     self.gate_temp.requires_grad_(True)


            ### default
            # converged = (loss_target < 1e-6) or (i > (self._iter*0.5))

            # b = 1e-1 if converged else 1e-4
            # c = 1e-4 if converged else 0
            # loss = loss_target + b * loss_entropy + c * loss_temp + 1e-4 * loss_gate

            ### NEW
            converged = (loss_target < 1e-6)
            b = 1e-1 if converged else 0
            c = 1e-4 if converged else 0
            loss = loss_target + b * loss_entropy + c * loss_temp + 1e-4 * loss_gate


            loss.backward()
            opt.step()

            if i % 100 == 0 or i==self._iter-1:
                z_hat_hard = _eval(self.forward, _eval=True)
                loss_hard = _eval.metric(z_hat_hard, _eval.target)
                losses = f"\n loss_final: {loss_hard.item():.2e}, loss_target: {loss_target.item():.2e}, loss_entropy: {loss_entropy.item():.2e}, loss_gate: {loss_gate.item():.2e}\nloss_relmse_soft: {(loss_target.item()/(torch.var(_eval.target) + 1e-14)):.2e}"
                logger.info(f"\nIteration {i}: {self.detokenize()} {losses}")
                _eval.plot(z_hat_hard.detach(), z_hat.detach(), _eval.target, i)
            

        # #### diversity regularization
        # flat_items = out.view(-1, out.shape[-1]).T
        # norm_items = torch.norm(flat_items, dim=1, keepdim=True)
        # flat_norm_items = flat_items / (norm_items + 1e-8)
        # G = (flat_norm_items @ flat_norm_items.T)
        # # loss_orth = 1e-2 * torch.norm(G - torch.eye(G.shape[0], device=G.device), p='fro')
        # # loss_orth.backward(retain_graph=True)

        # ##### Hook for natural gradient
        # ## is too unstable, so not using

        # # if logits.requires_grad:
        # #       G = G.detach()
        # #     def natl_grad_hook(grad):
        # #         # return torch.linalg.lstsq(
        # #         #     G + 1e-6 * torch.eye(G.shape[0], device=G.device), 
        # #         #     grad
        # #         # )[0]
        # #         # ginv = torch.linalg.pinv(G + 1e-6 * torch.eye(G.shape[0], device=G.device))
        # #         # return ginv @ grad

        # #         # U,S,Vt = torch.linalg.svd(G)
        # #         # threshold = 1e-2
        # #         # inv_S = torch.where(S < threshold, 0.0, 1.0 / S)
        # #         # return Vt.T @ torch.diag(inv_S) @ U.T @ grad

        # #     self.weights.register_hook(natl_grad_hook)
            

        # #####



###########
# class Operator(nn.Module):
#     def __init__(self, token_id, token_init, tokens: List):
#         super().__init__()
#         self.ops = nn.ModuleList(tokens)

#         self.weights = nn.Parameter(0.01 * torch.rand(len(tokens)))
#         self.weights.data[token_id] = 1.0
        
#     def forward(self, stack, temp=1.0, num_candidates=16, noise_std=0.5, **kwargs):
#         # 1. Execute ops on current stack
#         raw_items = [op(stack, **kwargs) for op in self.ops]

#         # 2. Align candidate batch dim (dim 0) vs data batch dims (dim 1+)
#         items = []
#         for item in raw_items:
#             if not isinstance(item, torch.Tensor):
#                 item = torch.tensor(item)
#             # If tensor is from eval data (e.g., shape (120,)), prepend candidate dim 0 -> (1, 120)
#             if item.ndim > 0 and item.shape[0] != num_candidates:
#                 item = item.unsqueeze(0)
#             items.append(item)

#         # 3. Broadcast across shapes -> e.g. (16, 120)
#         target_shape = torch.broadcast_shapes(*[t.shape for t in items])
#         out = torch.stack([t.broadcast_to(target_shape) for t in items], dim=-1)

#         # 4. Langevin noisy weight sampling across B candidates
#         logits = torch.clamp(self.weights, min=-10.0, max=10.0)
#         logits = logits.unsqueeze(0).expand(num_candidates, -1) # (B, num_ops)
#         if noise_std > 0 and self.training:
#             logits = logits + torch.randn_like(logits) * noise_std

#         # 5. Mask NaNs & compute Softmax probabilities
#         has_nan = torch.stack([torch.isnan(t).any() for t in items])
#         logits = logits.masked_fill(has_nan.unsqueeze(0), float('-inf'))

#         safe_temp = max(float(temp), 0.01)
#         probs = F.softmax(logits / safe_temp, dim=-1) # (B, num_ops)

#         # Shape matching: (B, 1, ..., num_ops) to align with (*target_shape, num_ops)
#         view_shape = [num_candidates] + [1] * (out.ndim - 2) + [len(self.ops)]
#         probs_expanded = probs.view(*view_shape)

#         next_item = torch.sum(probs_expanded * torch.nan_to_num(out, 0.0), dim=-1)
#         return [*stack, next_item]

#     def get_token(self):
#         _id = torch.argmax(self.weights).item()
#         _value = self.ops[_id].value if hasattr(self.ops[_id], 'value') else None
#         return _id, _value


# class Stack(nn.Module):
#     def __init__(self, init_str: str, vocab: Vocab, **kwargs):
#         super().__init__()
#         self.vocab = vocab
#         self.ops = nn.ModuleList(self.vocab.tokenize_str(init_str))
#         self.max_depth = kwargs.get("max_depth", 12)

#         self.temp = nn.Parameter(20 * torch.ones(self.vocab.seq_len))
#         self.output_logits = nn.Parameter(torch.zeros(min(self.vocab.seq_len, self.max_depth)))
#         self.output_logits.data[-1] = 1.0
#         self.output_temp = 1.0

#     def forward(self, stack_in=None, num_candidates=16, noise_std=0.5, **kwargs):
#         # Shape (B, 1) so candidate dim 0 is explicit
#         if stack_in is None:
#             device = self.output_logits.device
#             stack_in = [torch.zeros(num_candidates, 1, device=device)] * self.vocab.seq_len

#         for i, op in enumerate(self.ops):
#             stack_in = op(
#                 stack_in, 
#                 temp=self.temp[i], 
#                 num_candidates=num_candidates, 
#                 noise_std=noise_std, 
#                 **kwargs
#             )[-self.max_depth:]

#         # Stack depth outputs: shape (B, N, depth)
#         elems = torch.stack(stack_in[-self.vocab.seq_len:], dim=-1)
#         attn_weights = F.softmax(self.output_logits / self.output_temp, dim=0)
        
#         # Output shape: (B, N)
#         out = torch.sum(elems * attn_weights, dim=-1)
#         return out

#     def loss(self, z_hat, z_target, metric):
#         # z_hat: (B, N), z_target: (N,) -> broadcasts to (B, N)
#         loss_target = metric(z_hat, z_target).mean()
#         loss_temp = (self.temp).pow(2.0).mean()
#         return loss_target, loss_temp

#     @torch.no_grad()
#     def detokenize(self):
#         tokens = []
#         for op in self.ops:
#             token_id, value = op.get_token()
#             if token_id == self.vocab.scalar_token_id and value is not None:
#                 tokens.append(f"{value.item():.4f}")
#             else:
#                 tokens.append(self.vocab.id_to_token[token_id])
#         return " ".join(tokens).replace('<unk>', '').strip()

#     def _train(self, _eval, num_candidates=16, noise_std=.5):
#         opt = torch.optim.Adam(self.parameters(), lr=1e-1)
#         for i in range(1000):
#             opt.zero_grad()
            
#             z_hat = _eval(lambda **kw: self.forward(num_candidates=num_candidates, noise_std=noise_std, **kw))
#             loss_target, loss_temp = self.loss(z_hat, _eval.target, _eval.metric)
            
#             loss = loss_target + 1e-4 * loss_temp 
#             loss.backward()
#             opt.step()

#             if i % 100 == 0:
#                 print(f"Step {i} | Loss: {loss_target.item():.4f} | Program: {self.detokenize()}")