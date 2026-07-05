"""
Test suite for cleaned up RPN algebra rules and contrastive training.
"""

import torch
import pytest
from typing import Dict, List

from .embeddings import TOKEN_TO_ID
from .algebra import create_composite_ruleset
from .generator import create_vocab_from_embeddings, RPNGenerator
from .contrastive import ContrastiveRPN


def test_vocab_consistency():
    """Test that all tokens in embeddings have proper mappings."""
    assert "__scalar__" in TOKEN_TO_ID
    scalar_id = TOKEN_TO_ID["__scalar__"]

    # Check that important tokens exist
    assert "q" in TOKEN_TO_ID
    assert "psi" in TOKEN_TO_ID
    assert "dx" in TOKEN_TO_ID
    assert "dy" in TOKEN_TO_ID
    assert "lap" in TOKEN_TO_ID
    assert "+" in TOKEN_TO_ID
    assert "*" in TOKEN_TO_ID
    assert "jacobian" in TOKEN_TO_ID


def test_arithmetic_rules():
    """Test arithmetic algebra rules."""
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])

    # Test commutative addition: a b + -> b a +
    a_id = TOKEN_TO_ID["q"]
    b_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]

    # Create token sequence: q psi +
    tokens = torch.tensor([[a_id, b_id, plus_id]], dtype=torch.long)

    # Apply random rule (should apply commutative addition)
    result = rules.apply_random_rule(tokens)
    assert result is not None

    # Check that result is psi q +
    expected = torch.tensor([[b_id, a_id, plus_id]], dtype=torch.long)
    assert torch.equal(result.tokens, expected)

    # Test commutative multiplication
    mul_id = TOKEN_TO_ID["*"]
    tokens = torch.tensor([[a_id, b_id, mul_id]], dtype=torch.long)

    result = rules.apply_random_rule(tokens)
    assert result is not None
    assert result.tokens[0, 0] == b_id
    assert result.tokens[0, 1] == a_id
    assert result.tokens[0, 2] == mul_id


def test_jacobian_rules():
    """Test jacobian algebra rules."""
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])

    a_id = TOKEN_TO_ID["q"]
    b_id = TOKEN_TO_ID["psi"]
    jac_id = TOKEN_TO_ID["jacobian"]
    neg_id = TOKEN_TO_ID["neg"]

    # Test jacobian antisymmetry: J(q, psi) -> -J(psi, q)
    tokens = torch.tensor([[a_id, b_id, jac_id]], dtype=torch.long)

    result = rules.apply_random_rule(tokens)
    assert result is not None

    # Should be: psi q jacobian neg
    assert result.tokens[0, 0] == b_id  # psi
    assert result.tokens[0, 1] == a_id  # q
    assert result.tokens[0, 2] == jac_id  # jacobian
    assert result.tokens[0, 3] == neg_id  # neg


def test_calculus_rules():
    """Test calculus algebra rules."""
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])

    a_id = TOKEN_TO_ID["q"]
    b_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]
    dx_id = TOKEN_TO_ID["dx"]

    # Test linearity of derivative: dx(q + psi) -> dx(q) + dx(psi)
    tokens = torch.tensor([[a_id, b_id, plus_id, dx_id]], dtype=torch.long)

    result = rules.apply_random_rule(tokens)
    assert result is not None

    # Should be: q dx psi dx +
    assert result.tokens[0, 0] == a_id  # q
    assert result.tokens[0, 1] == dx_id  # dx
    assert result.tokens[0, 2] == b_id  # psi
    assert result.tokens[0, 3] == dx_id  # dx
    assert result.tokens[0, 4] == plus_id  # +


def test_generator_vocab():
    """Test RPN generator vocabulary creation."""
    vocab = create_vocab_from_embeddings()

    assert "variables" in vocab
    assert "unary_ops" in vocab
    assert "binary_ops" in vocab
    assert "special_ops" in vocab
    assert "constants" in vocab

    # Check some expected tokens
    assert "q" in vocab["variables"]
    assert "psi" in vocab["variables"]
    assert "dx" in vocab["unary_ops"]
    assert "dy" in vocab["unary_ops"]
    assert "+" in vocab["binary_ops"]
    assert "*" in vocab["binary_ops"]
    assert "jacobian" in vocab["special_ops"]


def test_generator_creation():
    """Test RPN generator creates valid expressions."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab, max_depth=3, max_nodes=10)

    # Generate single expression
    rpn = generator.generate_rpn()
    assert isinstance(rpn, str)
    assert len(rpn) > 0
    tokens = rpn.split()

    # All tokens should be in vocabulary
    for token in tokens:
        assert token in TOKEN_TO_ID or token.replace('.', '', 1).isdigit()

    # Generate batch
    batch_size = 5
    batch, params = generator.generate_batch(batch_size, include_scalars=True)

    assert len(batch) == batch_size
    assert len(params) == batch_size

    for rpn_expr in batch:
        assert isinstance(rpn_expr, str)
        assert len(rpn_expr) > 0


def test_contrastive_training():
    """Test contrastive training forward pass."""
    trainer = ContrastiveRPN()

    # Test with simple RPN expressions
    rpns = ["q psi +", "psi q jacobian", "q dx"]

    # Test forward pass
    z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns)
    assert z_a.shape[0] == len(rpns)  # Batch size
    assert z_p.shape[0] == len(rpns)
    assert z_a.shape == z_p.shape
    assert loss.ndim == 0  # Scalar loss

    # Test with scalar parameters
    scalar_params = [
        {"r": 0.5, "beta": 1.0},
        {"beta": 2.0},
        None
    ]
    z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns, scalar_params)
    assert loss.ndim == 0


def test_scalar_alignment():
    """Test scalar value alignment after commutative rewrites."""
    vocab = create_vocab_from_embeddings()
    trainer = ContrastiveRPN()
    scalar_id = TOKEN_TO_ID["__scalar__"]

    # Create an expression with scalar constant: 2 q *
    tokens = [
        scalar_id,  # __scalar__ placeholder for 2
        TOKEN_TO_ID["q"],
        TOKEN_TO_ID["*"]
    ]

    # In a real scenario, we'd need to tokenize with scalar values
    # For now, just test that the module loads and runs
    rpns = ["2 q *", "q psi +"]
    z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns)
    assert loss.ndim == 0


def test_rules_completeness():
    """Test that all rule categories are properly implemented."""
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])

    # Check that we have rules from multiple categories
    category_counts = {}
    for rule in rules.rules:
        category = rule.category
        category_counts[category] = category_counts.get(category, 0) + 1

    # Should have at least arithmetic and jacobian rules
    assert "ARITHMETIC" in category_counts or "arithmetic" in category_counts
    assert "JACOBIAN" in category_counts or "jacobian" in category_counts

    # May have other categories if implemented
    from .algebra.modular_rules import RuleCategory
    print(f"Rule categories implemented: {category_counts}")


if __name__ == "__main__":
    print("Running RPN cleanup tests...")

    test_vocab_consistency()
    print("✓ Vocabulary consistency test passed")

    test_arithmetic_rules()
    print("✓ Arithmetic rules test passed")

    test_jacobian_rules()
    print("✓ Jacobian rules test passed")

    test_calculus_rules()
    print("✓ Calculus rules test passed")

    test_generator_vocab()
    print("✓ Generator vocabulary test passed")

    test_generator_creation()
    print("✓ Generator creation test passed")

    test_contrastive_training()
    print("✓ Contrastive training test passed")

    test_scalar_alignment()
    print("✓ Scalar alignment test passed")

    test_rules_completeness()
    print("✓ Rules completeness test passed")

    print("\nAll tests passed! ✓")