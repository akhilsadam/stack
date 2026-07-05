"""
Calculus derivative / linearity equivalence rules.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_calculus_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Linearity of derivative operators: dx(a+b) = dx(a) + dx(b), etc."""
    rule_set = AlgebraicRuleSet("calculus", vocab, pad_token_id=pad_token_id)

    # Helper to match derivative operator tokens
    derivative_ops = {"dx", "dy", "lap"}

    def match_derivative_linearity(op_token: str):
        def match_fn(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
            B, L = token_ids.shape
            if L < 4:  # Need op + at least 2 tokens for a + b op
                return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

            op_id = vocab.get(op_token, -1)
            return token_ids[:, -1] == op_id

        return match_fn

    def linearity_transform(op_token: str):
        def transform_fn(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
            """``… a b + dx`` -> ``… a dx b dx +`` (and similarly for dy, lap)"""
            B, L = token_ids.shape
            prefix = token_ids[:, :-4]
            a = token_ids[:, -4:-3]
            b = token_ids[:, -3:-2]
            plus_op = token_ids[:, -2:-1]
            deriv_op = token_ids[:, -1:]

            # Check that middle operator is actually +
            plus_id = vocab.get("+", -1)
            if not (plus_op == plus_id).all():
                return token_ids.clone()

            # Apply linearity: dx(a+b) = dx(a) + dx(b)
            result = torch.cat([
                prefix,
                a,
                deriv_op,
                b,
                deriv_op,
                plus_op
            ], dim=-1)
            return result

        return transform_fn

    # Add linearity rules for each derivative operator
    for op in ["dx", "dy"]:
        rule_set.add_rule(
            SimpleAlgebraicRule(
                name=f"linearity_{op}",
                category=RuleCategory.CALCULUS,
                description=f"{op}(a+b) = {op}(a) + {op}(b)",
                pattern_length=4,
                output_length=6,
                match_fn=match_derivative_linearity(op),
                transform_fn=linearity_transform(op),
            )
        )

    # Special handling for laplacian (lap) - same pattern
    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="linearity_lap",
            category=RuleCategory.CALCULUS,
            description="lap(a+b) = lap(a) + lap(b)",
            pattern_length=4,
            output_length=6,
            match_fn=match_derivative_linearity("lap"),
            transform_fn=linearity_transform("lap"),
        )
    )

    # Chain rule for derivatives of products: dx(a*b) = a*dx(b) + b*dx(a)
    def match_product_chain_rule(op_token: str):
        def match_fn(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
            B, L = token_ids.shape
            if L < 4:
                return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

            op_id = vocab.get(op_token, -1)
            mul_id = vocab.get("*", -1)
            return (token_ids[:, -1] == op_id) & (token_ids[:, -2] == mul_id)

        return match_fn

    def product_chain_rule_transform(op_token: str):
        def transform_fn(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
            """``… a b * dx`` -> ``… a b dx * b a dx * +``"""
            B, L = token_ids.shape
            prefix = token_ids[:, :-4]
            a = token_ids[:, -4:-3]
            b = token_ids[:, -3:-2]
            mul_op = token_ids[:, -2:-1]
            deriv_op = token_ids[:, -1:]

            mul_id = vocab.get("*", -1)
            plus_id = vocab.get("+", -1)

            # Ensure we have a multiplication operator
            if not (mul_op == mul_id).all():
                return token_ids.clone()

            # dx(a*b) = a*dx(b) + b*dx(a)
            result = torch.cat([
                prefix,
                a,
                b,
                deriv_op,
                b.clone(),
                a.clone(),
                deriv_op.clone(),
                mul_op.new_full((B, 1), mul_id),
                mul_op.new_full((B, 1), plus_id)
            ], dim=-1)
            return result

        return transform_fn

    for op in ["dx", "dy"]:
        rule_set.add_rule(
            SimpleAlgebraicRule(
                name=f"product_rule_{op}",
                category=RuleCategory.CALCULUS,
                description=f"{op}(a*b) = a*{op}(b) + b*{op}(a)",
                pattern_length=4,
                output_length=9,
                match_fn=match_product_chain_rule(op),
                transform_fn=product_chain_rule_transform(op),
            )
        )

    return rule_set