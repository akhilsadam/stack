"""
Logarithm identities on RPN token ID tensors.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_logarithm_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Logarithm identities: ln(ab)=ln(a)+ln(b), ln(1)=0, etc."""
    rule_set = AlgebraicRuleSet("logarithm", vocab, pad_token_id=pad_token_id)

    # We need a ln token - check if it exists in vocab, otherwise use placeholder
    if "ln" not in vocab and "log" not in vocab:
        # No logarithm token in vocabulary, return empty rule set
        return rule_set

    ln_token = "ln" if "ln" in vocab else "log"
    ln_id = vocab.get(ln_token, -1)

    # ln(a*b) = ln(a) + ln(b)
    def match_ln_product(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 4:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        mul_id = vocab.get("*", -1)
        return (token_ids[:, -1] == ln_id) & (token_ids[:, -2] == mul_id)

    def ln_product_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a b * ln`` -> ``… a ln b ln +``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-4]
        a = token_ids[:, -4:-3]
        b = token_ids[:, -3:-2]
        mul_op = token_ids[:, -2:-1]

        plus_id = vocab.get("+", -1)

        result = torch.cat([
            prefix,
            a,
            mul_op.new_full((B, 1), ln_id),
            b,
            mul_op.new_full((B, 1), ln_id),
            mul_op.new_full((B, 1), plus_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="ln_product",
            category=RuleCategory.LOGARITHM,
            description="ln(a*b) = ln(a) + ln(b)",
            pattern_length=4,
            output_length=6,
            match_fn=match_ln_product,
            transform_fn=ln_product_transform,
        )
    )

    # ln(1) = 0 (special scalar constant rule)
    def match_ln_one(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 2:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        scalar_id = vocab.get("__scalar__", -1)
        return (token_ids[:, -1] == ln_id) & (token_ids[:, -2] == scalar_id)

    def ln_one_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… 1 ln`` -> ``… 0`` (assuming scalar 1)"""
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
            name="ln_one",
            category=RuleCategory.LOGARITHM,
            description="ln(1) = 0",
            pattern_length=2,
            output_length=1,
            match_fn=match_ln_one,
            transform_fn=ln_one_transform,
        )
    )

    # Inverse relationship with exp: ln(exp(a)) = a
    def match_ln_exp(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        exp_id = vocab.get("exp", -1)
        return (token_ids[:, -1] == ln_id) & (token_ids[:, -2] == exp_id)

    def ln_exp_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a exp ln`` -> ``… a``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-3]
        a = token_ids[:, -3:-2]

        result = torch.cat([
            prefix,
            a
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="ln_exp",
            category=RuleCategory.LOGARITHM,
            description="ln(exp(a)) = a",
            pattern_length=3,
            output_length=1,
            match_fn=match_ln_exp,
            transform_fn=ln_exp_transform,
        )
    )

    return rule_set