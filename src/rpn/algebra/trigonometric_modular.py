"""
Trigonometric identities on RPN token ID tensors.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_trigonometric_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Trigonometric identities: sin^2+cos^2=1, angle addition formulas, etc."""
    rule_set = AlgebraicRuleSet("trigonometric", vocab, pad_token_id=pad_token_id)

    # sin^2 + cos^2 = 1
    def match_sin2_cos2_1(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 7:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        sin_id = vocab.get("sin", -1)
        cos_id = vocab.get("cos", -1)
        square_id = vocab.get("square", -1)
        plus_id = vocab.get("+", -1)

        # Match pattern: sin square cos square +
        return (
            (token_ids[:, -1] == plus_id) &
            (token_ids[:, -2] == square_id) &
            (token_ids[:, -3] == cos_id) &
            (token_ids[:, -4] == square_id) &
            (token_ids[:, -5] == sin_id)
        )

    def sin2_cos2_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a sin square a cos square +`` -> ``… 1``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-7]

        scalar_id = vocab.get("__scalar__", -1)
        result = torch.cat([
            prefix,
            token_ids.new_full((B, 1), scalar_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="sin2_plus_cos2_equals_1",
            category=RuleCategory.TRIGONOMETRIC,
            description="sin^2(a) + cos^2(a) = 1",
            pattern_length=7,
            output_length=1,
            match_fn=match_sin2_cos2_1,
            transform_fn=sin2_cos2_transform,
        )
    )

    # cos(-a) = cos(a)
    def match_cos_even(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        cos_id = vocab.get("cos", -1)
        neg_id = vocab.get("neg", -1)
        return (token_ids[:, -1] == cos_id) & (token_ids[:, -2] == neg_id)

    def cos_even_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a neg cos`` -> ``… a cos``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-3]
        a = token_ids[:, -3:-2]

        result = torch.cat([
            prefix,
            a,
            token_ids[:, -1:]
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="cos_even",
            category=RuleCategory.TRIGONOMETRIC,
            description="cos(-a) = cos(a)",
            pattern_length=3,
            output_length=2,
            match_fn=match_cos_even,
            transform_fn=cos_even_transform,
        )
    )

    # sin(-a) = -sin(a)
    def match_sin_odd(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        sin_id = vocab.get("sin", -1)
        neg_id = vocab.get("neg", -1)
        return (token_ids[:, -1] == sin_id) & (token_ids[:, -2] == neg_id)

    def sin_odd_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a neg sin`` -> ``… a sin neg``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-3]
        a = token_ids[:, -3:-2]

        result = torch.cat([
            prefix,
            a,
            token_ids[:, -1:],
            token_ids[:, -2:-1]  # neg operator
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="sin_odd",
            category=RuleCategory.TRIGONOMETRIC,
            description="sin(-a) = -sin(a)",
            pattern_length=3,
            output_length=3,
            match_fn=match_sin_odd,
            transform_fn=sin_odd_transform,
        )
    )

    # Double angle formulas (cos(2a) = 2cos^2(a)-1, sin(2a)=2sin(a)cos(a))
    # These would be complex to implement as simple suffix rules, so we'll leave them
    # as placeholders for now

    return rule_set