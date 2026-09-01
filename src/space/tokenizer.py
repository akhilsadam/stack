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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



# from .arity import get_tree_coords

# TODO extend to multidimensionality / matrix / vector / tensor equations
## needs to add physical dimension to token
## needs a PhysicalTensor class to extract tensor constants
## needs to handle tensor contractions and maybe Einstein summation notation...
## quite difficult for now as solver doesn't support either yet

# TokenSchema
# Generalized scalar evaluation lookups
# UNARY_MATH_OPS = {
#     "sin": math.sin,
#     "cos": math.cos,
#     "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
#     "square": lambda x: x**2,
#     "cube": lambda x: x**3,
#     "neg": lambda x: -x,
#     "exp": math.exp,
#     "log": lambda x: math.log(x) if x > 0 else float("nan"),
# }

# BINARY_MATH_OPS = {
#     "+": lambda a, b: a + b,
#     "-": lambda a, b: a - b,
#     "*": lambda a, b: a * b,
#     "/": lambda a, b: a / b if b != 0 else float("nan"),
# }

# def _is_zero(val: Optional[float]) -> bool:
#     return val is not None and math.isclose(val, 0.0, abs_tol=1e-6)


# def _is_one(val: Optional[float]) -> bool:
#     return val is not None and math.isclose(val, 1.0, abs_tol=1e-6)


# @dataclass
# class Rule:
#     """User-defined rewrite rule using RPN pattern matching.

#     Capitalized tokens (e.g., 'A', 'X') act as wildcard variables matching any subtree.
#     """

#     pattern: str  # e.g., "psi lap" or "A invlap lap"
#     replacement: str  # e.g., "q" or "A"



# class Node:
#     """Atomic syntax tree element."""

#     def __init__(
#         self, name: str, arity: int, value: Optional[float] = None
#     ):
#         self.name = name
#         self.arity = arity
#         self.value = value
#         self.children: List["Node"] = []

#     def clone(self) -> "Node":
#         new_node = Node(self.name, self.arity, self.value)
#         new_node.children = [c.clone() for c in self.children]
#         return new_node

#     @property
#     def is_scalar(self) -> bool:
#         return self.value is not None


# class Tree:
#     """Tree container for traversal, evaluation, pattern matching, and rule application."""

#     def __init__(self, root: Node):
#         self.root = root

#     def to_rpn(self) -> List[str]:
#         return self._to_rpn_node(self.root)

#     def _to_rpn_node(self, node: Node) -> List[str]:
#         rpn = []
#         for child in node.children:
#             rpn.extend(self._to_rpn_node(child))
#         rpn.append(
#             f"{node.value:.4g}" if node.value is not None else node.name
#         )
#         return rpn

#     @classmethod
#     def from_rpn(cls, tokens: List[str], vocab: "Vocab") -> Optional["Tree"]:
#         stack: List[Node] = []
#         for tok in tokens:
#             if not tok or tok in ("<unk>", "<pad>"):
#                 continue

#             # Wildcard tokens (e.g., 'A', 'X') during pattern parsing
#             if tok[0].isupper():
#                 stack.append(Node(tok, arity=0))
#                 continue

#             rep = vocab.rep_from_str(tok)
#             if rep._id == vocab.scalar_token_id:
#                 node = Node("<scalar>", 0, value=rep._mag)
#             else:
#                 arity = vocab.id_to_arity.get(rep._id, 0)
#                 node = Node(vocab.id_to_token[rep._id], arity)

#             if node.arity == 0:
#                 stack.append(node)
#             elif node.arity == 1:
#                 if len(stack) < 1:
#                     return None
#                 node.children = [stack.pop()]
#                 stack.append(node)
#             elif node.arity == 2:
#                 if len(stack) < 2:
#                     return None
#                 right = stack.pop()
#                 left = stack.pop()
#                 node.children = [left, right]
#                 stack.append(node)

#         return cls(stack[0]) if len(stack) == 1 else None

#     def simplify(self, vocab: "Vocab") -> "Tree":
#         self.root = self._simplify_node(self.root, vocab)
#         return self

#     def _simplify_node(self, node: Node, vocab: "Vocab") -> Node:
#         # Bottom-up recursion across children
#         node.children = [
#             self._simplify_node(c, vocab) for c in node.children
#         ]

#         # 1. User Rewrite Rules
#         for pat_tree, repl_tree in vocab.compiled_rules:
#             bindings: Dict[str, Node] = {}
#             if self._match(node, pat_tree.root, bindings):
#                 rewritten = self._instantiate(repl_tree.root, bindings)
#                 return self._simplify_node(rewritten, vocab)

#         # 2. Generalized Constant Folding & Unary/Binary Math Evaluation
#         if vocab.fold_constants:
#             # Unary evaluations (sin, cos, sqrt, square, cube, neg, exp, log)
#             if node.arity == 1 and len(node.children) == 1:
#                 child = node.children[0]
#                 if child.is_scalar and node.name in UNARY_MATH_OPS:
#                     try:
#                         res = UNARY_MATH_OPS[node.name](child.value)
#                         if not math.isnan(res) and not math.isinf(res):
#                             return Node("<scalar>", 0, value=res)
#                     except (ValueError, OverflowError, ZeroDivisionError):
#                         pass

#             # Binary evaluations (+, -, *, /)
#             elif node.arity == 2 and len(node.children) == 2:
#                 left, right = node.children[0], node.children[1]

#                 if (
#                     left.is_scalar
#                     and right.is_scalar
#                     and node.name in BINARY_MATH_OPS
#                 ):
#                     try:
#                         res = BINARY_MATH_OPS[node.name](
#                             left.value, right.value
#                         )
#                         if not math.isnan(res) and not math.isinf(res):
#                             return Node("<scalar>", 0, value=res)
#                     except (ValueError, OverflowError, ZeroDivisionError):
#                         pass

#                 # Associative Constant Merging: c1 * (c2 * X) -> (c1 * c2) * X
#                 if node.name == "*":
#                     if (
#                         left.is_scalar
#                         and right.name == "*"
#                         and len(right.children) == 2
#                     ):
#                         if right.children[0].is_scalar:
#                             merged = left.value * right.children[0].value
#                             new_node = Node("*", 2)
#                             new_node.children = [
#                                 Node("<scalar>", 0, value=merged),
#                                 right.children[1],
#                             ]
#                             return self._simplify_node(new_node, vocab)
#                         elif right.children[1].is_scalar:
#                             merged = left.value * right.children[1].value
#                             new_node = Node("*", 2)
#                             new_node.children = [
#                                 Node("<scalar>", 0, value=merged),
#                                 right.children[0],
#                             ]
#                             return self._simplify_node(new_node, vocab)

#                 # Tolerance-based Identity Reductions
#                 if node.name == "*":
#                     if _is_one(right.value):
#                         return left
#                     if _is_one(left.value):
#                         return right
#                     if _is_zero(right.value) or _is_zero(left.value):
#                         return Node("<scalar>", 0, value=0.0)
#                 elif node.name == "+":
#                     if _is_zero(right.value):
#                         return left
#                     if _is_zero(left.value):
#                         return right
#                 elif node.name == "-":
#                     if _is_zero(right.value):
#                         return left

#         return node

#     def _match(
#         self, node: Node, pattern: Node, bindings: Dict[str, Node]
#     ) -> bool:
#         if pattern.name[0].isupper():
#             if pattern.name in bindings:
#                 return self._to_rpn_node(node) == self._to_rpn_node(
#                     bindings[pattern.name]
#                 )
#             bindings[pattern.name] = node
#             return True

#         if node.name != pattern.name or len(node.children) != len(
#             pattern.children
#         ):
#             return False

#         return all(
#             self._match(c, pc, bindings)
#             for c, pc in zip(node.children, pattern.children)
#         )

#     def _instantiate(
#         self, template: Node, bindings: Dict[str, Node]
#     ) -> Node:
#         if template.name[0].isupper():
#             return bindings[template.name].clone()

#         new_node = Node(template.name, template.arity, template.value)
#         new_node.children = [
#             self._instantiate(c, bindings) for c in template.children
#         ]
#         return new_node
eps = 1e-6

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
        token = name.strip().lower()
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


class Affine(nn.Module):
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__()

        init = torch.atanh(torch.clamp(torch.tensor(scale) / 10.0, -0.999, 0.999)) * 10
        self.raw_scale = nn.Parameter(init)
        init = torch.atanh(torch.clamp(torch.tensor(shift) / 10.0, -0.999, 0.999)) * 10
        self.raw_shift = nn.Parameter(init)
    
    def forward(self, _in):
        return torch.tanh(self.raw_scale / 10.0) * 10.0 * _in + torch.tanh(self.raw_shift / 10.0) * 10.0

class Operator(nn.Module):
    def __init__(self, token_id, token_init, gate, tokens: List[Token]):
        super().__init__()

        ops = [tok() for tok in tokens] # default initialization

        n = len(tokens)
        # self.weights = nn.Parameter(torch.ones(n) / n)
        self.weights = nn.Parameter(torch.rand(n))

        
        if token_id >= 0: 
            self.weights.data.fill_(0.0)
            self.weights.data[token_id] = 1.0
            ops[token_id] = token_init

        self.ops = nn.ModuleList(ops)

        # self.postprocess = nn.ModuleList([Affine() for _ in range(n)])
        # self.weights.data = F.softmax(self.weights.data, dim=0)

        self.gate = nn.Parameter(torch.tensor(gate))

    def term_entropy(self, probs, n_active):
        return -(probs * torch.log(probs + eps)).sum() / torch.log(n_active.float() + eps)
        
    def p(self, items, temp):
        has_nan = torch.stack([torch.isnan(t).any() for t in items])
        n_active = (~has_nan).sum().clamp(min=1.0)

        logits = self.weights
        logits = logits.masked_fill(has_nan, float('-inf'))
        logits = logits / temp
        p = F.softmax(logits, dim=0)

        entropy = self.term_entropy(p, n_active)
        return p, entropy

    def forward(self, stack, temp=1.0, gate_temp=1.0, hardness=0.0, **kwargs):
        items = [(op(stack, **kwargs)) for i, op in enumerate(self.ops)]

        # items = [self.postprocess[i](op(stack, **kwargs)) for i, op in enumerate(self.ops)]

        # broadcast to largest shape
        target_shape = torch.broadcast_shapes(*[t.shape for t in items])
        items = [t.broadcast_to(target_shape) for t in items]
        out = torch.stack(items, dim=-1)

        p, term_entropy = self.p(items, temp)
        soft_item = torch.sum(p * out, dim=-1)
        hard_item = items[torch.argmax(p, dim=0)]

        # STE
        next_item = soft_item + hardness * (hard_item - soft_item).detach()

        # passthrough gate
        a = torch.sigmoid(self.gate / gate_temp)
        next_item = (1 - a) * next_item + a * stack[-1]

        return [*stack, next_item], term_entropy

    def get_token(self, gate_temp=1.0):
        _id = torch.argmax(self.weights).item()
        _value = self.ops[_id].value()
        return _id, _value, torch.sigmoid(self.gate / gate_temp)

class Vocab:

    def __init__(
        self,
        tokens: List[Token],
        seq_len: int = 64,
        _Operator = Operator,
        **kwargs,
    ):
        self.seq_len = seq_len
        # self.tokens = [UNK, *tokens]
        self.tokens = tokens
        self.token_to_id = {
            token.token: i for i, token in enumerate(self.tokens)
        }
        self.id_to_token = {
            i: token.token for i, token in enumerate(self.tokens)
        }
        self.id_to_arity = {
            i: token.arity for i, token in enumerate(self.tokens)
        }

        # self.pad_token_id = self.token_to_id["<unk>"]
        self._Operator = _Operator

    def __len__(self):
        return len(self.tokens)

    def rep_from_str(self, token_str: str) -> int:
        _id = self.token_to_id.get(token_str.strip().lower(), -1)
        return _id 

    def tokenize_str(self, _str: str):
        ops = []
        toks = _str.split(" ")
        for i, s in enumerate(toks):
            _id = self.rep_from_str(s)
            if _id == -1:
                t = None
                passthrough = 1.0
            else:
                passthrough = 1.0 # regardless we use passthrough to start from nothing
                t = self.tokens[_id]()
            ops.append(self._Operator(_id, t, passthrough, self.tokens))

        if len(toks) < self.seq_len:
            _id = -1 # self.pad_token_id
            ops.extend([self._Operator(_id, None, 1.0, self.tokens) for _ in range(self.seq_len - len(toks))])
        return ops

class Stack(nn.Module):
    def __init__(self, init_str: str, vocab: Vocab, T0=20, **kwargs):
        super().__init__()
        self.vocab = vocab

        self.ops = nn.ModuleList(
            self.vocab.tokenize_str(init_str)
        )

        self.base_stack = [torch.tensor(1.0),] * self.vocab.seq_len # padding
        self.max_depth = kwargs.get("max_depth", 12)

        self.T0 = T0
        self.temp = nn.Parameter(torch.ones(self.vocab.seq_len))
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

        self.output_logits = nn.Parameter(torch.zeros(min(self.vocab.seq_len, self.max_depth)))
        self.output_logits.data[-1] = 1.0
        self.output_temp = 1.0

        self._iter = kwargs.get('_iter', 2000)

    def T(self, temp, scale=1.0):
        return scale * F.sigmoid(temp) + eps + 0.01

    def reset_accum(self):
        self._term_entropy = torch.tensor(0.0)

    def forward(self, stack_in=None, _eval=False, **kwargs):
        stack_in = stack_in or self.base_stack
        temp = self.T(self.temp, self.T0)
        gate_temp = self.T(self.gate_temp)

        if _eval:
            gate_temp = self.T(self.gate_temp, 0.0)
            kwargs['hardness'] = 1.0

        term_entropy = 0.0
        for i, op in enumerate(self.ops):
            stack_in, _term_entropy = op(stack_in, temp = temp[i], gate_temp = gate_temp, **kwargs)
            # stack_in = stack_in[-self.max_depth:]
            term_entropy = term_entropy + _term_entropy

        # elems = torch.stack(stack_in[-self.vocab.seq_len:], dim=-1)
        # attn_weights = F.softmax(self.output_logits / self.output_temp, dim=0)
        # out = torch.sum(elems * attn_weights, dim=-1) # provides gradient, bypass constants that gradient block
        # return out
        
        # assign cached vars
        self._term_entropy = self._term_entropy + term_entropy

        return stack_in[-1] # last element

    @torch.no_grad()
    def detokenize(self, threshold = 0.95):
        tokens = []
        gates = []
        final = []
        arity = []

        gate_temp = self.T(self.gate_temp)
        for i, op in enumerate(self.ops):
            token_id, value, gate = op.get_token(gate_temp)
            tokens.append(self.vocab.id_to_token[token_id])
            gates.append(gate)
            if gate < threshold:
                final.append(tokens[-1])
                arity.append(self.vocab.id_to_arity[token_id])

        return '\n'*2 + '\t' +\
         '\n\t'.join([" ".join(tokens), 
                     ' '.join([f'{g:.2f}' for g in gates]), 
                     " ".join(final).strip()]) + \
         '\n'*2

            
    
    def _train(self, _eval):
        opt = torch.optim.Adam(self.parameters(), lr=1e-1)
        for i in range(self._iter):
            opt.zero_grad()

            self.reset_accum()
            z_hat = _eval(self.forward)

            # log schedule
            T_schedule = math.exp(-i/200)

            loss_target = _eval.metric(z_hat, _eval.target)
            loss_temp = F.relu(self.T(self.temp) - T_schedule).pow(2.0).mean()
            loss_entropy = self._term_entropy
            loss_gate = self.T(self.gate_temp).pow(2.0).mean()
            # print(loss_target.item(), loss_temp.item(), self.gate_temp.item())
            logger.debug(f"Loss target: {loss_target.item():.2e}, loss_entropy: {loss_entropy.item():.2e}, temp: {F.sigmoid(self.temp).mean().item():.2e}, loss gate: {loss_gate.item():.2e}")


            # # loss = loss_target + 1e-4 * loss_temp 
            # # loss = max(loss_target, loss_temp / 20)
            # a = (loss_temp / 20) > loss_target
            # b = 0.05 if a else 1e-4 
            # c = 0 #1e-4
            # loss = loss_target + b * loss_temp + c * loss_gate

            b = 1e-1 if loss_target < 1e-1 else 1e-4
            loss = loss_target  + b *  loss_entropy + 1e-4 * loss_temp
            # loss = max(loss_target, loss_entropy + 1e-1)
            

            if (loss_target > 1e-3 and loss_entropy > 1e-2):
                # not converged yet; don't optimize gate
                self.gate_temp.requires_grad_(False)
            else:
                self.gate_temp.requires_grad_(True)

            loss.backward()
            opt.step()
            if i % 100 == 0 or i==self._iter-1:
                # print(i, self.detokenize())
                losses = f"\nloss_target: {loss_target.item():.2e}, loss_entropy: {loss_entropy.item():.2e}, loss_gate: {loss_gate.item():.2e}"
                logger.info(f"\nIteration {i}: {self.detokenize()} {losses}")
                z_hat_hard = _eval(self.forward, _eval=True)
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