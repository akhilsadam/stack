"""
Comprehensive tests for RPN generator.
"""

import torch
import pytest
from typing import Dict, List

from qg.solver.opt.operator.rpn.generator import (
    RPNGenerator,
    create_vocab_from_embeddings,
    NodeType,
    Arity,
)
from qg.solver.opt.operator.rpn.embeddings import TOKEN_TO_ID


def test_vocab_creation():
    """Test create_vocab_from_embeddings."""
    vocab = create_vocab_from_embeddings()

    # Check all required categories exist
    required_categories = ["variables", "unary_ops", "binary_ops", "special_ops", "constants"]
    for cat in required_categories:
        assert cat in vocab
        assert isinstance(vocab[cat], list)

    # Check some expected tokens in categories
    assert "q" in vocab["variables"]
    assert "psi" in vocab["variables"]
    assert "dx" in vocab["unary_ops"] or "dx" in vocab.get("linear_ops", [])
    assert "+" in vocab["binary_ops"]
    assert "*" in vocab["binary_ops"]
    assert "jacobian" in vocab["special_ops"] or "jacobian" in vocab.get("binary_ops", [])
    assert "__scalar__" in vocab["constants"]

    # Check no empty categories (except constants which might be just ["__scalar__"])
    for cat, tokens in vocab.items():
        if cat != "constants":  # constants can be just ["__scalar__"]
            assert len(tokens) > 0, f"Category {cat} is empty"


def test_generator_initialization():
    """Test RPNGenerator initialization."""
    vocab = create_vocab_from_embeddings()

    # Test with default parameters
    gen1 = RPNGenerator(vocab)
    assert gen1.max_depth == 4
    assert gen1.max_nodes == 20
    assert gen1.constant_prob == 0.2
    assert hasattr(gen1, 'operator_distribution')
    assert hasattr(gen1, 'operator_arity')
    assert hasattr(gen1, 'operator_vocab')

    # Test with custom parameters
    gen2 = RPNGenerator(
        vocab,
        max_depth=2,
        max_nodes=10,
        constant_prob=0.5,
        operator_distribution={"+": 0.7, "*": 0.3}
    )
    assert gen2.max_depth == 2
    assert gen2.max_nodes == 10
    assert gen2.constant_prob == 0.5
    assert "+" in gen2.operator_distribution
    assert "*" in gen2.operator_distribution
    # Distribution should be normalized
    total_prob = sum(gen2.operator_distribution.values())
    assert abs(total_prob - 1.0) < 1e-6


def test_node_type_enum():
    """Test NodeType enum."""
    assert NodeType.VARIABLE.value == "variable"
    assert NodeType.CONSTANT.value == "constant"
    assert NodeType.OPERATOR.value == "operator"

    # Test instantiation
    var_node = NodeType.VARIABLE
    assert isinstance(var_node, NodeType)


def test_arity_enum():
    """Test Arity enum."""
    assert Arity.UNARY.value == 1
    assert Arity.BINARY.value == 2
    assert Arity.SPECIAL.value == 3

    # Test instantiation
    unary = Arity.UNARY
    assert isinstance(unary, Arity)


def test_operator_info_building():
    """Test operator info is built correctly."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab)

    # Check that operator_arity is populated
    assert len(gen.operator_arity) > 0

    # Check some known operators
    if "+" in gen.operator_arity:
        assert gen.operator_arity["+"] == Arity.BINARY
    if "*" in gen.operator_arity:
        assert gen.operator_arity["*"] == Arity.BINARY
    if "dx" in gen.operator_arity:
        assert gen.operator_arity["dx"] == Arity.UNARY

    # Check operator_vocab mapping
    for op, vocab_cat in gen.operator_vocab.items():
        assert vocab_cat in ["unary_ops", "binary_ops", "special_ops"]
        assert op in vocab[vocab_cat]


def test_random_operator_selection():
    """Test random operator selection respects distribution."""
    vocab = create_vocab_from_embeddings()

    # Create generator with simple distribution
    gen = RPNGenerator(
        vocab,
        operator_distribution={"+": 0.6, "*": 0.4}
    )

    # Sample many times
    samples = [gen._random_operator() for _ in range(1000)]

    # Check distribution roughly matches
    plus_count = samples.count("+")
    mul_count = samples.count("*")

    # Should be roughly 60/40 split
    plus_ratio = plus_count / 1000
    assert 0.55 < plus_ratio < 0.65  # Allow some randomness


def test_expression_tree_generation():
    """Test expression tree generation."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab, max_depth=3, max_nodes=8)

    # Generate a tree
    tree, node_count = gen.generate_expression_tree()

    # Check tree structure
    assert isinstance(tree, dict)
    assert "type" in tree
    assert tree["type"] in [NodeType.VARIABLE, NodeType.CONSTANT, NodeType.OPERATOR]

    if tree["type"] == NodeType.OPERATOR:
        assert "value" in tree
        assert "children" in tree
        assert isinstance(tree["children"], list)
        assert len(tree["children"]) > 0

        # Check children are valid
        for child in tree["children"]:
            assert isinstance(child, dict)
            assert "type" in child

    elif tree["type"] in [NodeType.VARIABLE, NodeType.CONSTANT]:
        assert "value" in tree
        # Value should be a string
        assert isinstance(tree["value"], str)

    # Node count should be within bounds
    assert 0 < node_count <= gen.max_nodes


def test_tree_to_rpn_conversion():
    """Test converting expression tree to RPN."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab)

    # Test leaf nodes
    var_tree = {"type": NodeType.VARIABLE, "value": "q"}
    rpn_var = gen.tree_to_rpn(var_tree)
    assert rpn_var == ["q"]

    const_tree = {"type": NodeType.CONSTANT, "value": "2.5"}
    rpn_const = gen.tree_to_rpn(const_tree)
    assert rpn_const == ["2.5"]

    # Test unary operator
    unary_tree = {
        "type": NodeType.OPERATOR,
        "value": "dx",
        "children": [{"type": NodeType.VARIABLE, "value": "q"}]
    }
    rpn_unary = gen.tree_to_rpn(unary_tree)
    assert rpn_unary == ["q", "dx"]

    # Test binary operator
    binary_tree = {
        "type": NodeType.OPERATOR,
        "value": "+",
        "children": [
            {"type": NodeType.VARIABLE, "value": "q"},
            {"type": NodeType.VARIABLE, "value": "psi"}
        ]
    }
    rpn_binary = gen.tree_to_rpn(binary_tree)
    assert rpn_binary == ["q", "psi", "+"]


def test_generate_rpn_single():
    """Test generating single RPN expression."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab, max_depth=3, max_nodes=6)

    # Generate single expression
    rpn = gen.generate_rpn()

    assert isinstance(rpn, str)
    assert len(rpn) > 0

    # Split and check tokens
    tokens = rpn.split()
    assert len(tokens) > 0

    # All tokens should be valid
    for token in tokens:
        # Check if it's a number
        try:
            float(token)
            is_number = True
        except ValueError:
            is_number = False

        # Token should either be a number or in vocabulary
        if not is_number:
            # Check against TOKEN_TO_ID
            # Note: token might be normalized (e.g., "nabla" instead of "∇")
            # So we check normalization
            from qg.solver.opt.operator.rpn.embeddings import normalize_token
            normalized = normalize_token(token)
            assert normalized in TOKEN_TO_ID or normalized == "__scalar__"


def test_generate_rpn_multiple():
    """Test generating multiple RPN expressions."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab, max_depth=2, max_nodes=4)

    # Generate 5 expressions
    n = 5
    rpns = gen.generate_rpn(n)

    assert isinstance(rpns, list)
    assert len(rpns) == n

    for rpn in rpns:
        assert isinstance(rpn, str)
        assert len(rpn) > 0

        # Quick validation
        tokens = rpn.split()
        assert len(tokens) <= gen.max_nodes * 2  # Rough upper bound


def test_generate_batch():
    """Test batch generation."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab, max_depth=2, max_nodes=5)

    # Test without scalar params
    batch_size = 3
    rpns = gen.generate_batch(batch_size)
    assert len(rpns) == batch_size
    
    batch_size = 8
    # Test with scalar params
    rpns2 = gen.generate_batch(batch_size)
    assert len(rpns2) == batch_size

def test_expression_complexity_limits():
    """Test that generated expressions respect complexity limits."""
    vocab = create_vocab_from_embeddings()

    # Test with very tight limits
    gen = RPNGenerator(vocab, max_depth=1, max_nodes=2)

    for _ in range(10):  # Generate multiple times
        rpn = gen.generate_rpn()
        tokens = rpn.split()

        # With max_depth=1 and max_nodes=2, expressions should be very simple
        # Typically just a variable or constant, or maybe a unary operation
        assert len(tokens) <= 3  # Very conservative bound


def test_random_variable_selection():
    """Test random variable selection."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab)

    # Sample variables many times
    samples = [gen._random_variable() for _ in range(100)]

    # All samples should be in variables list
    for sample in samples:
        assert sample in vocab["variables"]

    # Should get different variables (not guaranteed but likely)
    unique_samples = set(samples)
    assert len(unique_samples) > 1 or len(vocab["variables"]) == 1


def test_random_constant_generation():
    """Test random constant generation."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab)

    # Sample constants many times
    samples = [gen._random_constant() for _ in range(50)]

    for sample in samples:
        # Should be convertible to float
        try:
            val = float(sample)
            assert isinstance(val, float)
        except ValueError:
            pytest.fail(f"Constant '{sample}' is not a valid float")

        # Check range (based on implementation)
        val = float(sample)
        assert 0 <= val <= 10 or 0.1 <= val <= 5.0


def test_jacobian_special_handling():
    """Test special handling for jacobian operator."""
    vocab = create_vocab_from_embeddings()
    gen = RPNGenerator(vocab)

    # Check if jacobian is in special handling
    if "jacobian" in gen.operator_arity:
        # Jacobian should have special arity handling
        # It might be Arity.BINARY or Arity.SPECIAL
        assert gen.operator_arity["jacobian"] in [Arity.BINARY, Arity.SPECIAL]


def test_deterministic_with_seed():
    """Test that generation can be deterministic with seed."""
    import random

    vocab = create_vocab_from_embeddings()

    # Set random seed
    random.seed(42)
    torch_seed = torch.manual_seed(42)

    gen1 = RPNGenerator(vocab, max_depth=2, max_nodes=4)
    expr1 = gen1.generate_rpn()

    # Reset and use same seed
    random.seed(42)
    torch.manual_seed(42)

    gen2 = RPNGenerator(vocab, max_depth=2, max_nodes=4)
    expr2 = gen2.generate_rpn()

    # Should be identical with same seed
    assert expr1 == expr2


def test_vocab_completeness():
    """Test that generator vocabulary covers all needed tokens."""
    vocab = create_vocab_from_embeddings()

    # Generate many expressions
    gen = RPNGenerator(vocab)
    expressions = [gen.generate_rpn() for _ in range(100)]

    # Collect all tokens used
    all_tokens = set()
    for expr in expressions:
        tokens = expr.split()
        for token in tokens:
            # Normalize for comparison
            from qg.solver.opt.operator.rpn.embeddings import normalize_token
            norm_token = normalize_token(token)
            all_tokens.add(norm_token)

    # Check that all used tokens are either "__scalar__" or in TOKEN_TO_ID
    for token in all_tokens:
        if token != "__scalar__":
            assert token in TOKEN_TO_ID, f"Token '{token}' not in vocabulary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])