"""Integration tests for the RPN tokenizer stack against the ``qg`` package.

These tests wire three pieces together:

1. :class:`space.tokens.TokenEmbedding` — RPN string <-> continuous embedding.
2. ``qg``'s :class:`RPNCompiler` — compiles an RPN string into a callable that
   evaluates the expression on a quasi-geostrophic flow state (the semantic
   ``_eval`` a :class:`tokenizer.Tokenizer` scores against).
3. :class:`arch.layer.MLP` + :class:`arch.flow.NF` — the MLP-Mixer coupling maps
   used by the normalizing flow, checked for shape-preserving invertibility.

The ``qg`` solver is the source of truth for what an RPN expression *means*, so
we compile the same vocabulary the token embedder uses and evaluate it on a
real spectral state.
"""

from typing import List

import pytest
import torch

from apps.qg._eval import QGEvaluator, build_vocab, make_batch_strings
from space.tokens import Token, TokenEmbedding, Vocab
from arch.layer import MLP
from arch.flow import NF


# ---------------------------------------------------------------------------
# qg-backed semantic evaluation
# ---------------------------------------------------------------------------

def test_qg_evaluates_vocab_expressions():
    """Every expression in the batch compiles and evaluates to a finite field."""
    evaluator = QGEvaluator(grid_size=32)
    strings = make_batch_strings()

    values = evaluator(strings)

    assert values.shape == (len(strings), 32, 32)
    assert torch.isfinite(values).all()


def test_qg_distinguishes_distinct_expressions():
    """Semantically different RPNs map to different field values."""
    evaluator = QGEvaluator(grid_size=32)

    q = evaluator.eval_one("q")
    q_scaled = evaluator.eval_one("q 1.25 *")
    jac = evaluator.eval_one("q psi jacobian")

    # q 1.25 * is exactly 1.25 * q on the same state.
    assert torch.allclose(q_scaled, 1.25 * q, atol=1e-5)
    # The jacobian is a genuinely different (nonlinear) field.
    assert not torch.allclose(jac, q, atol=1e-3)


# ---------------------------------------------------------------------------
# TokenEmbedding round-trip (RPN string <-> embedding)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [12])
@pytest.mark.parametrize("embed_dim", [8])
def test_token_embedding_round_trip_is_invertible(seq_len: int, embed_dim: int):
    torch.manual_seed(7)

    vocab = build_vocab()
    embedder = TokenEmbedding(vocab=vocab, seq_len=seq_len, embed_dim=embed_dim, phys_dim=1)
    embedder.eval()  # deterministic (no dequantization noise)

    batch_strings = make_batch_strings()
    embeddings = embedder(batch_strings)
    decoded = embedder.reverse(embeddings)

    assert decoded == batch_strings


def test_embedded_expressions_are_qg_evaluable():
    """Decoded RPN strings survive the round-trip AND remain valid ``qg`` PDEs."""
    torch.manual_seed(7)

    vocab = build_vocab()
    embedder = TokenEmbedding(vocab=vocab, seq_len=12, embed_dim=8, phys_dim=1)
    embedder.eval()
    evaluator = QGEvaluator(grid_size=32)

    strings = make_batch_strings()
    decoded = embedder.reverse(embedder(strings))

    assert decoded == strings
    values = evaluator(decoded)
    assert torch.isfinite(values).all()


# ---------------------------------------------------------------------------
# MLP-Mixer layer + normalizing flow
# ---------------------------------------------------------------------------

def test_mlp_mixer_is_shape_preserving():
    """The MLP-Mixer maps (B, seq_len, dim) to the same shape."""
    torch.manual_seed(0)
    seq_len, dim, batch = 12, 8, 4

    mlp = MLP(seq_len, dim)
    x = torch.randn(batch, seq_len, dim)
    y = mlp(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mlp_mixer_mixes_across_sequence():
    """Token mixing means the output at one position depends on other positions.

    Perturbing a single sequence position should change the output at *other*
    positions — otherwise the layer is not a sequence-to-sequence mixer.
    """
    torch.manual_seed(0)
    seq_len, dim = 12, 8

    mlp = MLP(seq_len, dim)
    mlp.eval()

    x = torch.randn(1, seq_len, dim)
    y = mlp(x)

    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 1.0
    y_perturbed = mlp(x_perturbed)

    # Some position other than 0 must have changed.
    other_positions_changed = (y_perturbed[:, 1:, :] - y[:, 1:, :]).abs().max()
    assert other_positions_changed > 1e-4


def test_mlp_mixer_drives_normalizing_flow():
    """The mixer works as the coupling map inside NF and the flow is invertible."""
    torch.manual_seed(0)
    seq_len, dim, batch, steps = 12, 8, 4, 3

    f_z = MLP(seq_len, dim)
    f_n = MLP(seq_len, dim)
    f_z.eval()
    f_n.eval()

    x = torch.randn(batch, seq_len, dim)

    # Forward coupling then exact inverse should recover the input.
    a, b = NF.map_fwd(x, x, f_z, f_n, steps)
    a_rec, b_rec = NF.map_bwd(a, b, f_z, f_n, steps)

    assert a.shape == x.shape and b.shape == x.shape
    assert torch.allclose(a_rec, x, atol=1e-4)
