import math
import random
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any
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

from .opt.GNA import GradNormAdam

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

        passthrough = 1.0
        # passthrough = 0.5
        
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

        # self.output_logits = nn.Parameter(torch.zeros(min(self.vocab.seq_len, self.capacity)))
        # self.output_logits.data[-1] = 1.0
        # self.output_temp = 1.0

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
        mux_val = []

        if _eval:
            gate_temp = self.T(self.gate_temp, 0.0)
            kwargs['hardness'] = 1.0

        _entropy = 0.0
        for i, op in enumerate(self.ops):
            stack, _term_entropy, _gate_entropy, p = op(stack, temp = temp[i], gate_temp = gate_temp, **kwargs)
            # stack_in = stack_in[-self.max_depth:]
            _entropy = _entropy + _term_entropy + _gate_entropy / len(self.vocab.tokens)
            mux_val.append(p)


        # mux_val = torch.stack(mux_val, dim=0) # stack n [yx]

        # # should not be deficient rank
        # loss = logdet_si_loss(mux_val) * 15
        # loss.backward(retain_graph=True)

        # elems = stack[-self.vocab.seq_len:].to(torch.float32)
        # attn_weights = F.softmax(self.output_logits / self.output_temp, dim=0).to(torch.float32)
        # out = torch.einsum("s..., s -> ...", elems, attn_weights) # provides gradient, bypass constants that gradient block
        # return out
        
        # assign cached vars
        self._term_entropy = self._term_entropy + _entropy

        return stack[-1] # last element
        # out = stack[-1]
        # return out

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
        """Train the stack.

        Device/precision handling, metric logging, and artifact capture are
        all provided by the ambient environment (e.g. a
        ``with run_manager:`` block) — this method has no knowledge of them.
        """
        opt = create_optimizer(self, torch.optim.Adam, lr)

        for i in range(self._iter):
            opt.zero_grad()

            self.reset_accum()
            z_hat = _eval(self.forward)

            loss_target = _eval.metric(z_hat, _eval.target)  # full-batch

            loss_entropy = self._term_entropy
            loss_temp = self.T(self.temp).pow(2.0).mean()
            loss_gate = self.T(self.gate_temp).pow(2.0).mean()

            logger.debug(f"Loss target: {loss_target.item():.2e}, loss_entropy: {loss_entropy.item():.2e}, temp: {F.sigmoid(self.temp).mean().item():.2e}, loss gate: {loss_gate.item():.2e}")

            # NEW loss formulation
            converged = (loss_target < 1e-6)
            b = 0 # 1 if converged else 0
            c = 0 #1e-4 if converged else 0
            loss = loss_target + b * loss_entropy + c * loss_temp + 1e-4 * loss_gate

            loss.backward()

            opt.step()

            # Log metrics via RunManager
            if (i % 100 == 0 or i == self._iter - 1):
                z_hat_hard = _eval(self.forward, _eval=True)
                loss_hard = _eval.metric(z_hat_hard, _eval.target)

                logger.info("", extra={
                    "metrics": {
                        "loss_target": loss_target.item(),
                        "loss_hard": loss_hard.item(),
                        "loss_relmse": (loss_hard.item() / (torch.var(_eval.target) + 1e-14)).item(),
                        "loss_entropy": loss_entropy.item(),
                        "loss_gate": loss_gate.item(),
                        "temp_mean": F.sigmoid(self.temp).mean().item(),
                    },
                    "step": i,
                })

                losses = f"\nhard: loss_target: {loss_hard.item():.2e}, loss_relmse: {(loss_hard.item()/(torch.var(_eval.target) + 1e-14)):.2e}" \
                + f"\nsoft: loss_target: {loss_target.item():.2e}, loss_relmse: {(loss_target.item()/(torch.var(_eval.target) + 1e-14)):.2e}" \
                + f"\n      loss_entropy: {loss_entropy.item():.2e}, loss_gate: {loss_gate.item():.2e}"
                logger.info(f"\nIteration {i}: {self.detokenize()} {losses}")
                _eval.plot(z_hat_hard.detach(), z_hat.detach(), _eval.target, i)