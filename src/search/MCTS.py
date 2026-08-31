"""
Windowed Monte-Carlo Tree Search over the discrete token choices of a Stack.

Each Operator's existing softmax `weights` are used, unmodified, as the PUCT
prior at that position -- MCTS doesn't replace them, it searches *with* them
as a guide and returns a better (visit-count) distribution than the prior
alone would give. Scalar constants are still fit purely by gradient descent
(Adam on `Scalar.raw`), run locally inside leaf evaluation with everything
else held fixed -- structure search and constant fitting stay decoupled.

Leaf evaluation is exact: once a window's tokens are fixed, execution is a
single deterministic numeric pass (no softmax mixture), so there's no need
for a learned value net here -- real evaluation is already cheap.

Search covers a short, contiguous window of positions at a time (the full
seq_len is too large a tree for a modest simulation budget); `grow` slides
that window across the sequence, front-to-back or back-to-front.

Requires an `MCTSVocab` (below) rather than the base `Vocab`, since search
needs each Operator to expose a prior and support hard/committed execution.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from space.tokenizer import Operator, Scalar, Stack, Vocab


class PUCTOperator(Operator):
    """An Operator that can report its softmax weights as a search prior,
    execute a single hard-chosen token (no mixture), and lock in a winning
    token once search has picked one."""

    def prior(self) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.weights, dim=0)

    def execute(self, stack, token_id: int, tok=None, **kwargs):
        """Exact execution of one token, bypassing the mixture. `tok`
        overrides `self.ops[token_id]` -- used to pass in a private Scalar
        instance instead of the shared vocab singleton (see Node)."""
        tok = tok if tok is not None else self.ops[token_id]
        value = tok(stack, **kwargs)
        a = torch.sigmoid(self.gate)
        return (1 - a) * value + a * stack[-1]

    def commit(self, token_id: int, tok=None):
        """Lock this position onto `token_id`. Same weight convention as
        __init__ (a peak, not a one-hot), so ordinary training can keep
        sharpening it; `hardness` is flipped so forward is already hard."""
        if tok is not None:
            self.ops[token_id] = tok
        with torch.no_grad():
            self.weights.data.copy_(0.1 * torch.rand_like(self.weights))
            self.weights.data[token_id] = 1.0
        self.hardness = 1.0

class Node:
    """One edge of the search tree: "at this position, choose this token".

    `scalar`, when set, is a *private* Scalar() created for this edge alone
    -- the base Vocab's generic scalar slot is one shared instance across
    every position, so reusing it directly here would let unrelated window
    positions fight over the same parameter during search.
    """

    __slots__ = ("token_id", "scalar", "P", "N", "W", "children")

    def __init__(self, token_id: int = -1, prior: float = 0.0, scalar=None):
        self.token_id = token_id
        self.scalar = scalar
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.children: Dict[int, "Node"] = {}

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTSStack(Stack):
    """Stack with windowed PUCT search for committing discrete structure,
    on top of the ordinary gradient-based training already in the base
    class (untouched -- `_train` still works exactly as before)."""

    def __init__(self, init_str: str, vocab: Vocab, **kwargs):
        super().__init__(init_str, vocab, **kwargs)
        self.c_puct = kwargs.get("c_puct", 1.5)
        self.n_sims = kwargs.get("n_sims", 200)
        self.scalar_steps = kwargs.get("scalar_steps", 20)
        self.scalar_lr = kwargs.get("scalar_lr", 0.1)

    # ---- exact (mixture-free) execution, used only during search --------

    def _execute(self, overrides: Dict[int, Node], **kwargs):
        stack_in = list(self.base_stack)
        for i, op in enumerate(self.ops):
            node = overrides.get(i)
            token_id, tok = (node.token_id, node.scalar) if node is not None \
                else (int(torch.argmax(op.weights)), None)
            value = op.execute(stack_in, token_id, tok, **kwargs)
            stack_in = (stack_in + [value])[-self.max_depth:]

        return stack_in[-1]

    def _fit_scalars(self, overrides: Dict[int, Node], _eval) -> float:
        """Gradient-fit any private Scalars in `overrides`, all other
        structure held hard/fixed. Returns the resulting loss."""
        params = [n.scalar.raw for n in overrides.values() if n.scalar is not None]
        fn = lambda stack_in=None, **kw: self._execute(overrides, **kw)
        if params:
            opt = torch.optim.Adam(params, lr=self.scalar_lr)
            for _ in range(self.scalar_steps):
                opt.zero_grad()
                loss = _eval.metric(_eval(fn), _eval.target)
                loss.backward()
                opt.step()
        with torch.no_grad():
            return _eval.metric(_eval(fn), _eval.target).item()

    # ---- tree search ------------------------------------------------------

    def _expand(self, node: Node, positions: List[int], depth: int):
        i = positions[depth]
        op = self.ops[i]
        prior = op.prior()
        for token_id in range(len(op.ops)):
            scalar = Scalar(0.0) if token_id == self.vocab.scalar_token_id else None
            node.children[token_id] = Node(token_id, prior[token_id].item(), scalar)

    def _select(self, node: Node) -> Tuple[int, Node]:
        total_N = sum(c.N for c in node.children.values())

        def score(c: Node) -> float:
            return c.Q + self.c_puct * c.P * math.sqrt(total_N + 1) / (1 + c.N)

        token_id = max(node.children, key=lambda k: score(node.children[k]))
        return token_id, node.children[token_id]

    def _simulate(self, root: Node, positions: List[int], _eval):
        node, path, depth = root, [root], 0
        overrides: Dict[int, Node] = {}

        while node.children and depth < len(positions):
            token_id, node = self._select(node)
            overrides[positions[depth]] = node
            path.append(node)
            depth += 1

        if depth < len(positions):
            self._expand(node, positions, depth)
            token_id, node = self._select(node)
            overrides[positions[depth]] = node
            path.append(node)
            depth += 1

        # cheap rollout for the rest of the window: sample each remaining
        # position from its prior. Never sample scalar here -- it's a
        # throwaway completion, not worth minting a private parameter for.
        with torch.no_grad():
            while depth < len(positions):
                i = positions[depth]
                prior = self.ops[i].prior().clone()
                prior[self.vocab.scalar_token_id] = 0.0
                token_id = int(torch.multinomial(prior + 1e-8, 1))
                overrides[i] = Node(token_id)
                depth += 1

        value = -self._fit_scalars(overrides, _eval)
        for n in path:
            n.N += 1
            n.W += value

    def search(self, start: int, width: int, _eval, n_sims: Optional[int] = None) -> float:
        """Run PUCT search over positions [start, start+width) and commit
        the most-visited path found. Returns the achieved loss."""
        positions = list(range(start, min(start + width, len(self.ops))))
        root = Node()
        self._expand(root, positions, 0)

        for _ in range(n_sims or self.n_sims):
            self._simulate(root, positions, _eval)

        node, overrides = root, {}
        for i in positions:
            if not node.children:
                break
            token_id = max(node.children, key=lambda k: node.children[k].N)
            node = node.children[token_id]
            overrides[i] = node
            self.ops[i].commit(node.token_id, node.scalar)

        return self._fit_scalars(overrides, _eval)

    def grow(self, _eval, width: int = 6, stride: Optional[int] = None,
             reverse: bool = False, n_sims: Optional[int] = None) -> float:
        """Slide a search window across the whole sequence, committing each
        window before moving to the next. `reverse=True` grows the program
        back-to-front (from the output end) instead of front-to-back."""
        stride = stride or width
        starts = list(range(0, len(self.ops), stride))
        if reverse:
            starts = list(reversed(starts))

        loss = None
        for start in starts:
            loss = self.search(start, width, _eval, n_sims)
            print(start, self.detokenize()[0])
        return loss

    def _train(self, _eval):
        """Complete training: grow the whole sequence via windowed PUCT
        search first (committing discrete structure, fitting scalars as it
        goes), then hand off to the base class's ordinary gradient descent
        (unchanged) to polish scalars/gates/temp/output-attention once the
        structure is in place. Same entry point and shape as `Stack._train`
        -- one call does the complete search-and-train."""
        self.window = 2
        self.stride = 1
        self.reverse = False
        self.grow(_eval, width=self.window, stride=self.stride, reverse=self.reverse)
        super()._train(_eval)