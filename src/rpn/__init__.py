"""
RPN (Reverse Polish Notation) PDE compiler and learned token embeddings.

Overview
~~~~~~~~

This module provides:

1. **Compiler** (:mod:`qg.solver.opt.operator.rpn.compiler`)
   - Parses RPN token sequences into executable PDE components
   - Maintains linear/nonlinear separation for spectral solver

2. **Embeddings** (:mod:`qg.solver.opt.operator.rpn.embeddings`)
   - Fixed vocabulary mapping token→ID→learned vector
   - Category-aware embedding (variable, operator, constant, …)
   - Scalar‑valued constants via small MLP (no discretisation)

3. **Algebra rules** (:mod:`qg.solver.opt.operator.rpn.algebra`)
   - Sound mathematical rewrites on token‑ID tensors
   - Used for contrastive‑learning positives (InfoNCE)
   - Modular: arithmetic, Jacobian, calculus, trigonometric, etc.

4. **Contrastive training** (:mod:`qg.solver.opt.operator.rpn.contrastive`)
   - Single‑encoder (RPN string → pooled embedding → projection)
   - Positive pairs from algebraic equivalence
   - Negative pairs from other batch sequences (InfoNCE)

5. **Generator** (:mod:`qg.solver.opt.operator.rpn.generator`)
   - Random RPN expression generator for training data
   - Configurable complexity and operator distribution
   - Supports scalar parameters and constants

All modules are designed for ``torch`` tensors and GPU acceleration.
"""

from .compiler import RPNCompiler
from .embeddings import (
    VOCAB_SIZE,
    SCALAR_TOKEN_ID,
    TOKEN_TO_ID,
    ID_TO_TOKEN,
    TOKEN_TO_CAT,
    normalize_token,
    batch_tokenize_rpn,
)
from .algebra import create_composite_ruleset
from .contrastive import ContrastiveRPN
from .qwen_contrastive import QwenContrastiveRPN

from .generator import RPNGenerator, create_vocab_from_embeddings

def batch_rpn_gen(batch_size, max_depth=25, max_nodes=50):
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab, max_depth=max_depth, max_nodes=max_nodes)
    return gen.generate_batch(batch_size)
 


# Re-export for cleaner API
__all__ = [
    # Embedding vocabulary
    "VOCAB_SIZE",
    "SCALAR_TOKEN_ID",
    "TOKEN_TO_ID",
    "ID_TO_TOKEN",
    "TOKEN_TO_CAT",
    "normalize_token",
    "batch_tokenize_rpn",
    # Algebra rules
    "create_composite_ruleset",
    # Contrastive learning
    "ContrastiveRPN",
    # Generator
    "RPNGenerator",
    "create_vocab_from_embeddings",
    "batch_rpn_gen"
]
