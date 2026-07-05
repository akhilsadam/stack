"""
Arithmetic equivalence rules on RPN token ID tensors.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_arithmetic_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Commutativity of ``+`` / ``*`` and double-negation elimination on the stack suffix."""
    rule_set = AlgebraicRuleSet("arithmetic", vocab, pad_token_id=pad_token_id)

    def match_binary_op(op_token: str):
        def match_fn(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
            B, L = token_ids.shape
            if L < 3:
                return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

            op_id = vocab.get(op_token, -1)
            return token_ids[:, -1] == op_id

        return match_fn

    def commutative_add_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a b +``  ->  ``… b a +``"""
        result = token_ids.clone()
        result[:, -3], result[:, -2] = token_ids[:, -2].clone(), token_ids[:, -3].clone()
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="commutative_addition",
            category=RuleCategory.ARITHMETIC,
            description="a + b = b + a",
            pattern_length=3,
            output_length=3,
            match_fn=match_binary_op("+"),
            transform_fn=commutative_add_transform,
        )
    )

    def commutative_mul_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a b *``  ->  ``… b a *``"""
        result = token_ids.clone()
        result[:, -3], result[:, -2] = token_ids[:, -2].clone(), token_ids[:, -3].clone()
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="commutative_multiplication",
            category=RuleCategory.ARITHMETIC,
            description="a * b = b * a",
            pattern_length=3,
            output_length=3,
            match_fn=match_binary_op("*"),
            transform_fn=commutative_mul_transform,
        )
    )

    def double_negation_match(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        neg_id = vocab.get("neg", -1)
        return (token_ids[:, -1] == neg_id) & (token_ids[:, -2] == neg_id)

    def double_negation_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a neg neg``  ->  ``… a``"""
        return token_ids[:, :-2]

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="double_negation",
            category=RuleCategory.ARITHMETIC,
            description="-(-a) = a",
            pattern_length=3,
            output_length=1,
            match_fn=double_negation_match,
            transform_fn=double_negation_transform,
        )
    )

    return rule_set
