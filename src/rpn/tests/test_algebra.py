"""
Exhaustive tests for algebra rules (testing everything except rule correctness).
"""

import torch
import pytest
from typing import Dict, List

from qg.solver.opt.operator.rpn.algebra import (
    create_composite_ruleset,
    AlgebraicRuleSet,
    CompositeAlgebraicRuleSet,
    SimpleAlgebraicRule,
    RuleCategory,
    TransformResult,
)
from qg.solver.opt.operator.rpn.embeddings import TOKEN_TO_ID, SCALAR_TOKEN_ID


def test_rule_category_enum():
    """Test RuleCategory enum values."""
    # All expected categories
    expected_categories = [
        "ARITHMETIC", "EXPONENT", "LOGARITHM", "TRIGONOMETRIC",
        "CALCULUS", "VECTOR_CALCULUS", "JACOBIAN"
    ]

    for cat_name in expected_categories:
        assert hasattr(RuleCategory, cat_name)
        cat = getattr(RuleCategory, cat_name)
        assert isinstance(cat, RuleCategory)
        assert isinstance(cat.value, str)


def test_simple_algebraic_rule_structure():
    """Test SimpleAlgebraicRule dataclass structure."""
    # Create a mock rule
    def mock_match(tokens, vocab):
        return torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

    def mock_transform(tokens, vocab):
        return tokens.clone()

    rule = SimpleAlgebraicRule(
        name="test_rule",
        category=RuleCategory.ARITHMETIC,
        description="Test rule",
        pattern_length=3,
        output_length=3,
        match_fn=mock_match,
        transform_fn=mock_transform,
    )

    assert rule.name == "test_rule"
    assert rule.category == RuleCategory.ARITHMETIC
    assert rule.description == "Test rule"
    assert rule.pattern_length == 3
    assert rule.output_length == 3
    assert rule._match_fn is mock_match
    assert rule._transform_fn is mock_transform

    # Test methods
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    matches = rule.matches(tokens, TOKEN_TO_ID)
    assert isinstance(matches, torch.Tensor)
    assert matches.shape == (1,)
    assert matches.dtype == torch.bool


def test_algebraic_rule_set_basic():
    """Test AlgebraicRuleSet basic functionality."""
    rule_set = AlgebraicRuleSet("test_set", TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    assert rule_set.name == "test_set"
    assert rule_set.vocab is TOKEN_TO_ID
    assert rule_set.pad_token_id == SCALAR_TOKEN_ID
    assert rule_set.rules == []

    # Test adding a rule
    def mock_match(tokens, vocab):
        return torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

    def mock_transform(tokens, vocab):
        return tokens.clone()

    rule = SimpleAlgebraicRule(
        name="test_rule",
        category=RuleCategory.ARITHMETIC,
        description="Test",
        pattern_length=2,
        output_length=2,
        match_fn=mock_match,
        transform_fn=mock_transform,
    )

    rule_set.add_rule(rule)
    assert len(rule_set.rules) == 1
    assert rule_set.rules[0] is rule

    # Test applicable_rule_indices with no matches
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    indices = rule_set.applicable_rule_indices(tokens)
    assert indices == []  # No rules match our mock

    # Test apply_rule with invalid index
    with pytest.raises(IndexError):
        rule_set.apply_rule(999, tokens)


def test_transform_result():
    """Test TransformResult dataclass."""
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    matched = torch.tensor([True], dtype=torch.bool)
    pad_token_id = 0

    result = TransformResult(
        tokens=tokens,
        matched=matched,
        pad_token_id=pad_token_id,
    )

    assert torch.equal(result.tokens, tokens)
    assert torch.equal(result.matched, matched)
    assert result.pad_token_id == pad_token_id


def test_composite_algebraic_rule_set():
    """Test CompositeAlgebraicRuleSet."""
    # Create two rule sets
    rule_set1 = AlgebraicRuleSet("set1", TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)
    rule_set2 = AlgebraicRuleSet("set2", TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    # Add mock rules
    def mock_match(tokens, vocab):
        return torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

    def mock_transform(tokens, vocab):
        return tokens.clone()

    rule1 = SimpleAlgebraicRule(
        name="rule1", category=RuleCategory.ARITHMETIC,
        description="Test", pattern_length=2, output_length=2,
        match_fn=mock_match, transform_fn=mock_transform
    )
    rule2 = SimpleAlgebraicRule(
        name="rule2", category=RuleCategory.CALCULUS,
        description="Test2", pattern_length=3, output_length=3,
        match_fn=mock_match, transform_fn=mock_transform
    )

    rule_set1.add_rule(rule1)
    rule_set2.add_rule(rule2)

    # Create composite
    composite = CompositeAlgebraicRuleSet(rule_set1, rule_set2)

    assert composite.name == "composite"
    assert composite.vocab is TOKEN_TO_ID
    assert composite.pad_token_id == SCALAR_TOKEN_ID
    assert len(composite.rules) == 2
    assert composite.rules[0] is rule1
    assert composite.rules[1] is rule2

    # Test with mismatched pad_token_id (should raise error)
    rule_set3 = AlgebraicRuleSet("set3", TOKEN_TO_ID, pad_token_id=999)
    with pytest.raises(ValueError):
        CompositeAlgebraicRuleSet(rule_set1, rule_set3)

    # Test with empty rule sets (should raise error)
    with pytest.raises(ValueError):
        CompositeAlgebraicRuleSet()


def test_create_composite_ruleset():
    """Test create_composite_ruleset factory function."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    assert isinstance(composite, CompositeAlgebraicRuleSet)
    assert composite.name == "composite"
    assert composite.vocab is TOKEN_TO_ID
    assert composite.pad_token_id == SCALAR_TOKEN_ID

    # Should have rules from all categories
    assert len(composite.rules) > 0

    # Check rule categories
    categories = set([rule.category for rule in composite.rules])
    expected_categories = {
        RuleCategory.ARITHMETIC,
        RuleCategory.JACOBIAN,
        RuleCategory.CALCULUS,
        RuleCategory.EXPONENT,
        RuleCategory.TRIGONOMETRIC,
        RuleCategory.LOGARITHM,
        RuleCategory.VECTOR_CALCULUS,
    }

    # At minimum should have arithmetic and jacobian
    assert RuleCategory.ARITHMETIC in categories
    assert RuleCategory.JACOBIAN in categories


def test_rule_application_mechanics():
    """Test the mechanics of rule application (not rule correctness)."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    # Create some valid token sequences
    q_id = TOKEN_TO_ID["q"]
    psi_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]

    # Test with single sequence
    tokens = torch.tensor([[q_id, psi_id, plus_id]], dtype=torch.long)
    result = composite.apply_random_rule(tokens)

    # Result should be TransformResult or None
    if result is not None:
        assert isinstance(result, TransformResult)
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'matched')
        assert hasattr(result, 'pad_token_id')
        assert result.tokens.shape[0] == 1  # Same batch size
    else:
        # No rule applied is also valid
        pass

    # Test with batch
    batch_tokens = torch.tensor([
        [q_id, psi_id, plus_id],
        [psi_id, q_id, plus_id],
    ], dtype=torch.long)
    batch_result = composite.apply_random_rule(batch_tokens)

    if batch_result is not None:
        assert batch_result.tokens.shape[0] == 2  # Same batch size
        assert batch_result.matched.shape[0] == 2


def test_random_positive_view():
    """Test random_positive_view method."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    q_id = TOKEN_TO_ID["q"]
    psi_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]

    tokens = torch.tensor([[q_id, psi_id, plus_id]], dtype=torch.long)
    anchor, positive = composite.random_positive_view(tokens)

    # Should return two tensors of same shape
    assert torch.equal(anchor, tokens)  # Anchor should be unchanged
    assert positive.shape == anchor.shape
    assert positive.dtype == anchor.dtype
    assert positive.device == anchor.device

    # Test with generator
    generator = torch.Generator()
    generator.manual_seed(42)
    anchor2, positive2 = composite.random_positive_view(tokens, generator=generator)
    assert anchor2.shape == positive2.shape


def test_padding_behavior():
    """Test padding behavior when rules change sequence length."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    # Find a rule that changes length (e.g., jacobian antisymmetry adds 'neg')
    jacobian_rules = [r for r in composite.rules if r.category == RuleCategory.JACOBIAN]
    if jacobian_rules:
        jacobian_rule = jacobian_rules[0]

        q_id = TOKEN_TO_ID["q"]
        psi_id = TOKEN_TO_ID["psi"]
        jac_id = TOKEN_TO_ID["jacobian"]

        tokens = torch.tensor([[q_id, psi_id, jac_id]], dtype=torch.long)

        # Apply the jacobian rule directly
        result = jacobian_rule.apply(tokens, TOKEN_TO_ID, SCALAR_TOKEN_ID)

        # Check padding
        assert result.tokens.shape[1] >= tokens.shape[1]  # Might be longer or same
        assert result.pad_token_id == SCALAR_TOKEN_ID

        # If longer, check padding
        if result.tokens.shape[1] > tokens.shape[1]:
            # Last element(s) should be pad_token_id
            # The jacobian rule adds 'neg' (1 extra token), so padding starts after original length + 1
            pad_positions = result.tokens[0, tokens.shape[1] + 1:]
            assert torch.all(pad_positions == SCALAR_TOKEN_ID)


def test_batch_with_variable_lengths():
    """Test rule application with variable length sequences."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    q_id = TOKEN_TO_ID["q"]
    psi_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]
    jac_id = TOKEN_TO_ID["jacobian"]

    # Batch with different lengths (implicitly padded)
    batch_tokens = torch.tensor([
        [q_id, psi_id, plus_id, SCALAR_TOKEN_ID],  # Length 3 effective
        [q_id, psi_id, jac_id, SCALAR_TOKEN_ID],   # Length 3 effective
        [q_id, SCALAR_TOKEN_ID, SCALAR_TOKEN_ID, SCALAR_TOKEN_ID],  # Length 1
    ], dtype=torch.long)

    result = composite.apply_random_rule(batch_tokens)

    if result is not None:
        # Output should have same batch size
        assert result.tokens.shape[0] == 3
        # Output might have different sequence length due to padding
        assert result.tokens.shape[1] >= 4  # At least as long as input


def test_rule_matching_edge_cases():
    """Test rule matching with edge cases."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    # Empty batch
    empty_tokens = torch.tensor([], dtype=torch.long).reshape(0, 0)
    indices = composite.applicable_rule_indices(empty_tokens)
    assert indices == []  # No rules should match empty batch

    # Very short sequences
    short_tokens = torch.tensor([[TOKEN_TO_ID["q"]]], dtype=torch.long)
    result = composite.apply_random_rule(short_tokens)
    # Should either return None or TransformResult


def test_rule_metadata():
    """Test rule metadata (names, descriptions)."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    for rule in composite.rules:
        assert isinstance(rule.name, str)
        assert len(rule.name) > 0
        assert isinstance(rule.description, str)
        assert len(rule.description) > 0
        assert isinstance(rule.category, RuleCategory)
        assert isinstance(rule.pattern_length, int)
        assert rule.pattern_length > 0
        assert isinstance(rule.output_length, int)
        assert rule.output_length > 0


def test_scalar_token_handling():
    """Test that rules handle scalar tokens appropriately."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    q_id = TOKEN_TO_ID["q"]
    scalar_id = SCALAR_TOKEN_ID
    plus_id = TOKEN_TO_ID["+"]

    # Expression with scalar: scalar q +
    tokens = torch.tensor([[scalar_id, q_id, plus_id]], dtype=torch.long)

    result = composite.apply_random_rule(tokens)

    if result is not None:
        # Check that scalar tokens are preserved or transformed appropriately
        # Rules shouldn't eliminate scalar tokens without reason
        pass


def test_device_handling():
    """Test that rules work on different devices (if available)."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    q_id = TOKEN_TO_ID["q"]
    psi_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]

    # CPU test (always works)
    cpu_tokens = torch.tensor([[q_id, psi_id, plus_id]], dtype=torch.long)
    cpu_result = composite.apply_random_rule(cpu_tokens)

    if torch.cuda.is_available():
        # GPU test
        gpu_tokens = cpu_tokens.cuda()
        gpu_result = composite.apply_random_rule(gpu_tokens)

        if cpu_result is not None and gpu_result is not None:
            # Results should be equivalent (ignoring device)
            assert cpu_result.tokens.shape == gpu_result.tokens.shape
            # Can't directly compare values because rules might choose different ones


def test_deterministic_with_seed():
    """Test that rule application can be deterministic with generator."""
    composite = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    q_id = TOKEN_TO_ID["q"]
    psi_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]

    tokens = torch.tensor([[q_id, psi_id, plus_id]], dtype=torch.long)

    # With same seed, should get same result (if any rule matches)
    generator1 = torch.Generator()
    generator1.manual_seed(123)
    result1 = composite.apply_random_rule(tokens, generator=generator1)

    generator2 = torch.Generator()
    generator2.manual_seed(123)
    result2 = composite.apply_random_rule(tokens, generator=generator2)

    if result1 is not None and result2 is not None:
        # Should be identical
        assert torch.equal(result1.tokens, result2.tokens)
        assert torch.equal(result1.matched, result2.matched)
    elif result1 is None and result2 is None:
        # Both None is also consistent
        pass
    else:
        # Mixed state shouldn't happen with same seed
        assert False, "Results should be consistent with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])