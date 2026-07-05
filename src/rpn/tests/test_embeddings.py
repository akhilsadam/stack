"""
Comprehensive tests for RPN token embeddings.
"""

import torch
import pytest
from typing import Dict, List

from qg.solver.opt.operator.rpn.embeddings import (
    VOCAB_SIZE,
    SCALAR_TOKEN_ID,
    TOKEN_TO_ID,
    ID_TO_TOKEN,
    TOKEN_TO_CAT,
    TokenCategory,
    normalize_token,
    batch_tokenize_rpn,
    RPNTokenEmbedder,
    tokenize_rpn,
)
from qg.solver.opt.operator.rpn.embeddings import _VOCAB_DEF  # Internal, but we test it


def test_vocab_structure():
    """Test vocabulary has proper structure and mappings."""
    # Check vocabulary size matches definition
    assert len(_VOCAB_DEF) == VOCAB_SIZE
    assert len(TOKEN_TO_ID) == VOCAB_SIZE
    assert len(ID_TO_TOKEN) == VOCAB_SIZE
    assert len(TOKEN_TO_CAT) == VOCAB_SIZE

    # Check scalar token exists
    assert "__scalar__" in TOKEN_TO_ID
    assert SCALAR_TOKEN_ID == TOKEN_TO_ID["__scalar__"]
    assert TOKEN_TO_CAT["__scalar__"] == TokenCategory.SCALAR_CONST

    # Check all tokens have consistent mappings
    for token, token_id in TOKEN_TO_ID.items():
        assert ID_TO_TOKEN[token_id] == token
        assert token in TOKEN_TO_CAT
        assert isinstance(TOKEN_TO_CAT[token], TokenCategory)

    # Check important tokens exist
    important_tokens = ["q", "psi", "dx", "dy", "lap", "+", "*", "jacobian"]
    for token in important_tokens:
        assert token in TOKEN_TO_ID
        assert token in TOKEN_TO_CAT

### TODO
# def test_normalize_token():
#     """Test token normalization function."""
#     # Test variable aliases
#     assert normalize_token("q") == "q"
#     assert normalize_token("Q") == "q"  # Lowercase
#     assert normalize_token(" omega ") == "q"  # Alias
#     assert normalize_token("ph") == "psi"  # Alias
#     assert normalize_token("PSI") == "psi"  # Lowercase

#     # Test operator aliases
#     assert normalize_token("∇") == "nabla"  # Unicode alias
#     assert normalize_token("nabla") == "nabla"
#     assert normalize_token("del") == "grad"  # grad alias
#     assert normalize_token("j") == "jacobian"  # Jacobian alias
#     assert normalize_token("mul") == "*"  # Multiplication alias

#     # Test case normalization
#     assert normalize_token("DX") == "dx"
#     assert normalize_token("Dy") == "dy"
#     assert normalize_token("LAP") == "lap"

#     # Test whitespace stripping
#     assert normalize_token("  q  ") == "q"
#     assert normalize_token("\tpsi\n") == "psi"


def test_tokenize_single_rpn():
    """Test single RPN tokenization."""
    # Test without scalar params
    tokens, values = tokenize_rpn("q psi +", None)
    assert tokens == ["q", "psi", "+"]
    assert values == [None, None, None]

    # Test with scalar literal
    tokens, values = tokenize_rpn("2 q *", None)
    assert tokens == ["__scalar__", "q", "*"]
    assert values == [2.0, None, None]

    # Test with named parameter
    tokens, values = tokenize_rpn("r psi *", {"r": 0.5})
    assert tokens == ["__scalar__", "psi", "*"]
    assert values == [0.5, None, None]

    # Test with both numeric and named params
    tokens, values = tokenize_rpn("2 r +", {"r": 1.5})
    assert tokens == ["__scalar__", "__scalar__", "+"]
    assert values == [2.0, 1.5, None]

    # Test missing named param (should warn but return 0.0)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tokens, values = tokenize_rpn("missing psi *", None)
        assert len(w) == 1
        assert "missing" in str(w[0].message)
        assert values[0] == 0.0


def test_batch_tokenize_rpn():
    """Test batch tokenization."""
    # Test simple batch
    rpns = ["q psi +", "q dx", "2 q *"]
    token_ids, scalar_vals, scalar_mask = batch_tokenize_rpn(rpns, [None, None, None])

    assert token_ids.shape[0] == 3  # Batch size
    assert scalar_vals.shape[0] == 3
    assert scalar_mask.shape[0] == 3

    # All batches should have same sequence length (padded)
    seq_len = token_ids.shape[1]
    assert scalar_vals.shape[1] == seq_len
    assert scalar_mask.shape[1] == seq_len

    # Test with scalar params
    scalar_params = [
        None,
        {"beta": 1.0},
        {"r": 0.5, "beta": 2.0}
    ]
    rpns = ["q psi +", "beta q *", "r psi beta +"]
    token_ids, scalar_vals, scalar_mask = batch_tokenize_rpn(rpns, scalar_params)

    # Check scalar mask identifies scalar tokens
    for i in range(3):
        row_mask = scalar_mask[i]
        row_ids = token_ids[i]
        for j in range(seq_len):
            if row_ids[j] == SCALAR_TOKEN_ID:
                assert row_mask[j] == True
            else:
                assert row_mask[j] == False


def test_embedder_initialization():
    """Test RPNTokenEmbedder initialization."""
    embedder = RPNTokenEmbedder(embed_dim=64, scalar_fourier=8)

    assert embedder.embed_dim == 64
    assert hasattr(embedder, 'token_emb')
    assert hasattr(embedder, 'scalar_emb')
    assert hasattr(embedder, 'out_norm')
    assert isinstance(embedder.out_norm, torch.nn.LayerNorm)


def test_embedder_forward():
    """Test RPNTokenEmbedder forward pass."""
    embedder = RPNTokenEmbedder(embed_dim=32, scalar_fourier=4)
    embedder.eval()  # Disable dropout if any

    # Create test batch
    batch_size = 2
    seq_len = 5

    # Mock token IDs (q=0, psi=1, +=2, scalar=some_id)
    q_id = TOKEN_TO_ID["q"]
    psi_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]
    scalar_id = SCALAR_TOKEN_ID

    # Batch 1: q psi + 2 *
    # Batch 2: 3 q +
    token_ids = torch.tensor([
        [q_id, psi_id, plus_id, scalar_id, plus_id],  # q psi + 2 +
        [scalar_id, q_id, plus_id, scalar_id, scalar_id]  # 3 q + 0 0 (padded)
    ], dtype=torch.long)

    scalar_vals = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    scalar_mask = torch.tensor([
        [False, False, False, True, False],
        [True, False, False, False, False]
    ], dtype=torch.bool)

    # Forward pass
    embeddings = embedder(token_ids, scalar_vals, scalar_mask)

    # Check output shape
    assert embeddings.shape == (batch_size, seq_len, 32)
    assert embeddings.dtype == torch.float32

    # Check no NaN or Inf
    assert torch.isfinite(embeddings).all()

    # Test with different sequence lengths
    rpns = ["q", "q psi +", "q psi dx *"]
    token_ids2, scalar_vals2, scalar_mask2 = batch_tokenize_rpn(rpns, [None, None, None])
    embeddings2 = embedder(token_ids2, scalar_vals2, scalar_mask2)

    assert embeddings2.shape[0] == 3
    assert embeddings2.shape[2] == 32  # embed_dim


def test_scalar_embedding():
    """Test ScalarEmbedding module."""
    from qg.solver.opt.operator.rpn.embeddings import ScalarEmbedding

    scalar_emb = ScalarEmbedding(embed_dim=32, n_fourier=8)

    # Test single value
    x = torch.tensor([1.5])
    out = scalar_emb(x)
    assert out.shape == (1, 32)

    # Test batch of values
    x_batch = torch.tensor([1.0, 2.0, 3.0])
    out_batch = scalar_emb(x_batch)
    assert out_batch.shape == (3, 32)

    # Test 2D input (batch, seq_len)
    x_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out_2d = scalar_emb(x_2d)
    assert out_2d.shape == (2, 2, 32)

    # Check no NaN
    assert torch.isfinite(out_2d).all()


def test_token_embedding():
    """Test TokenEmbedding module."""
    from qg.solver.opt.operator.rpn.embeddings import TokenEmbedding

    token_emb = TokenEmbedding(embed_dim=32)

    # Test single token ID
    token_ids = torch.tensor([TOKEN_TO_ID["q"]])
    out = token_emb(token_ids)
    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()

    # Test batch of token IDs
    token_ids = torch.tensor([
        TOKEN_TO_ID["q"],
        TOKEN_TO_ID["psi"],
        TOKEN_TO_ID["+"]
    ])
    out = token_emb(token_ids)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()

    # Test 2D input (batch, seq_len)
    token_ids = torch.tensor([
        [TOKEN_TO_ID["q"], TOKEN_TO_ID["psi"], TOKEN_TO_ID["+"]],
        [TOKEN_TO_ID["psi"], TOKEN_TO_ID["q"], TOKEN_TO_ID["*"]]
    ])
    out = token_emb(token_ids)
    assert out.shape == (2, 3, 32)
    assert torch.isfinite(out).all()

    ### TODO check later
    # Test scalar token (should return zeros)
    # scalar_ids = torch.tensor([[SCALAR_TOKEN_ID]])
    # out_scalar = token_emb(scalar_ids)
    # assert torch.allclose(out_scalar, torch.zeros_like(out_scalar))



def test_token_category_enum():
    """Test TokenCategory enum."""
    # Check all expected categories exist
    expected_categories = [
        "VARIABLE", "LINEAR_DIFF", "NONLINEAR_UNARY",
        "BINARY_OP", "VECTOR_OP", "JACOBIAN",
        "MISC_OP", "SCALAR_CONST"
    ]

    for cat_name in expected_categories:
        assert hasattr(TokenCategory, cat_name)
        cat = getattr(TokenCategory, cat_name)
        assert isinstance(cat, TokenCategory)

    # Check some example tokens have correct categories
    assert TOKEN_TO_CAT["q"] == TokenCategory.VARIABLE
    assert TOKEN_TO_CAT["dx"] == TokenCategory.LINEAR_DIFF
    assert TOKEN_TO_CAT["+"] == TokenCategory.BINARY_OP
    assert TOKEN_TO_CAT["sqrt"] == TokenCategory.NONLINEAR_UNARY
    assert TOKEN_TO_CAT["jacobian"] == TokenCategory.JACOBIAN
    assert TOKEN_TO_CAT["__scalar__"] == TokenCategory.SCALAR_CONST


def test_consistency_with_compiler():
    """Test that embeddings vocabulary matches compiler expectations."""
    # Variables that should be in vocabulary
    expected_variables = ["q", "psi", "u", "v", "x", "y"]
    for var in expected_variables:
        assert var in TOKEN_TO_ID
        assert TOKEN_TO_CAT[var] == TokenCategory.VARIABLE

    # Differential operators
    diff_ops = ["dx", "dy", "lap", "invlap"]
    for op in diff_ops:
        if op in TOKEN_TO_ID:  # invlap might not be present
            assert TOKEN_TO_CAT[op] == TokenCategory.LINEAR_DIFF

    # Binary operators
    binary_ops = ["+", "-", "*"]
    for op in binary_ops:
        assert op in TOKEN_TO_ID
        assert TOKEN_TO_CAT[op] == TokenCategory.BINARY_OP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])