import math
import random
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Token(nn.Module):
    """Token definition (by user)."""

    def __init__(self, token: str, arity: int, op: Callable = None):
        super().__init__()
        self.token = token.strip().lower()
        self.arity = arity
        self.op = op

class Scalar(Token):
    def __init__(self, value):
        super().__init__("<scalar>", 0, None)

        self.value = nn.Parameter(torch.tensor(value))
        self.op = self.forward
    
    def forward(self, stack):
        return self.value

class Operator(nn.Module):
    def __init__(self, token_id, token_init, tokens: List[Token]):
        super().__init__()
        self.ops = [token.op for token in tokens]
        self.ops[token_id] = token_init.op

        self.weights = nn.Parameter(0.01 * torch.rand(len(tokens)))
        self.weights.data[token_id] = 1.0
        
    def forward(self, stack, temp=1.0):
        out = torch.stack([op(stack) for op in self.ops], dim=-1)
        next_item = torch.sum(F.softmax(self.weights / temp, dim=0) * out, dim=-1)

        return [*stack, next_item]

class Vocab:

    def __init__(
        self,
        tokens: List[Token],
        seq_len: int
    ):
        self.tokens = [Token("<unk>", 1, lambda stack: 0.0), Token("<scalar>", 0, None), *tokens]
        self.token_to_id = {
            token.token: i for i, token in enumerate(self.tokens)
        }
        self.id_to_token = {
            i: token.token for i, token in enumerate(self.tokens)
        }
        self.id_to_arity = {
            i: token.arity for i, token in enumerate(self.tokens)
        }

        self.pad_token_id = self.token_to_id["<unk>"]
        self.scalar_token_id = self.token_to_id["<scalar>"]

    def __len__(self):
        return len(self.tokens)

    def rep_from_str(self, token_str: str) -> int:
        try: 
            val = float(token_str)
            return self.scalar_token_id, val
        except:
            _id = self.token_to_id.get(token_str.strip().lower(), self.pad_token_id)
            return _id, self.tokens[_id]

    def tokenize_str(self, _str: str):
        ops = []
        for i, s in enumerate(_str.split(" ")):
            _id, t = self.rep_from_str(s)
            ops.append(Operator(_id, t, self.tokens))
        if len(ops) < self.seq_len:
            _id = self.pad_token_id
            ops.extend([Operator(_id, self.tokens[_id], self.tokens) for _ in range(self.seq_len - len(ops))])
        return ops

class Stack(nn.Module):
    def __init__(self, init_str: str, vocab: Vocab, **kwargs):
        super().__init__()
        self.vocab = vocab

        self.ops = nn.ModuleList(
            self.vocab.tokenize_str(init_str)
        )

        self.base_stack = [0.0,] * self.vocab.seq_len # padding

        self.temp = nn.Parameter(torch.ones(self.vocab.seq_len))

    def forward(self, stack_in=None):
        stack_in = stack_in or self.base_stack

        for i, op in enumerate(self.ops):
            stack_in = op(stack_in, temp = self.temp[i])

        return stack_in[-1] # last element

    def loss(self, z_hat, z_target, metric):
        loss_target = metric(z_hat, z_target)
        loss_temp = (self.temp).pow(2.0).mean()
        print(loss_target.item(), loss_temp.item(), self.temp.mean().item())
        return loss_target, loss_temp
        