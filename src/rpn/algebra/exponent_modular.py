"""
Exponent / exp / power identities on RPN token ID tensors.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_exponent_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Exponential identities: exp(a+b)=exp(a)*exp(b), exp(0)=1, etc."""
    rule_set = AlgebraicRuleSet("exponent", vocab, pad_token_id=pad_token_id)

    # exp(a+b) = exp(a) * exp(b)
    def match_exp_sum(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 4:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        exp_id = vocab.get("exp", -1)
        plus_id = vocab.get("+", -1)
        return (token_ids[:, -1] == exp_id) & (token_ids[:, -2] == plus_id)

    def exp_sum_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a b + exp`` -> ``… a exp b exp *``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-4]
        a = token_ids[:, -4:-3]
        b = token_ids[:, -3:-2]
        plus_op = token_ids[:, -2:-1]
        exp_op = token_ids[:, -1:]

        mul_id = vocab.get("*", -1)

        result = torch.cat([
            prefix,
            a,
            exp_op.clone(),
            b,
            exp_op.clone(),
            b.new_full((B, 1), mul_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="exp_sum",
            category=RuleCategory.EXPONENT,
            description="exp(a+b) = exp(a)*exp(b)",
            pattern_length=4,
            output_length=6,
            match_fn=match_exp_sum,
            transform_fn=exp_sum_transform,
        )
    )

    # exp(a)*exp(b) = exp(a+b) - inverse rule
    def match_exp_mul(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 6:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        exp_id = vocab.get("exp", -1)
        mul_id = vocab.get("*", -1)
        return (token_ids[:, -1] == mul_id) & (token_ids[:, -2] == exp_id) & (token_ids[:, -4] == exp_id)

    def exp_mul_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a exp b exp *`` -> ``… a b + exp``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-6]
        a = token_ids[:, -6:-5]
        exp1 = token_ids[:, -5:-4]
        b = token_ids[:, -4:-3]
        exp2 = token_ids[:, -3:-2]
        mul = token_ids[:, -2:-1]

        plus_id = vocab.get("+", -1)
        exp_id = vocab.get("exp", -1)

        result = torch.cat([
            prefix,
            a,
            b,
            mul.new_full((B, 1), plus_id),
            mul.new_full((B, 1), exp_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="exp_mul",
            category=RuleCategory.EXPONENT,
            description="exp(a)*exp(b) = exp(a+b)",
            pattern_length=6,
            output_length=5,
            match_fn=match_exp_mul,
            transform_fn=exp_mul_transform,
        )
    )

    # exp(0) = 1 (special scalar constant rule)
    def match_exp_zero(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 2:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        exp_id = vocab.get("exp", -1)
        scalar_id = vocab.get("__scalar__", -1)
        return (token_ids[:, -1] == exp_id) & (token_ids[:, -2] == scalar_id)

    def exp_zero_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… 0 exp`` -> ``… 1`` (assuming scalar 0)"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-2]

        scalar_id = vocab.get("__scalar__", -1)
        result = torch.cat([
            prefix,
            token_ids.new_full((B, 1), scalar_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="exp_zero",
            category=RuleCategory.EXPONENT,
            description="exp(0) = 1",
            pattern_length=2,
            output_length=1,
            match_fn=match_exp_zero,
            transform_fn=exp_zero_transform,
        )
    )

    # Power laws: square = * self
    def match_square_expansion(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 2:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        square_id = vocab.get("square", -1)
        return token_ids[:, -1] == square_id

    def square_expansion_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a square`` -> ``… a a *``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-2]
        a = token_ids[:, -2:-1]

        mul_id = vocab.get("*", -1)
        result = torch.cat([
            prefix,
            a,
            a.clone(),
            a.new_full((B, 1), mul_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="square_expansion",
            category=RuleCategory.EXPONENT,
            description="square(a) = a*a",
            pattern_length=2,
            output_length=3,
            match_fn=match_square_expansion,
            transform_fn=square_expansion_transform,
        )
    )

    return rule_set