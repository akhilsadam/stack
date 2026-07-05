"""
Vector calculus identities on RPN token ID tensors.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_vector_calculus_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Vector calculus identities: div(grad)=lap, curl(curl)=grad(div)-lap, etc."""
    rule_set = AlgebraicRuleSet("vector_calculus", vocab, pad_token_id=pad_token_id)

    # div(grad) = laplacian
    def match_div_grad(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        div_id = vocab.get("div", -1)
        grad_id = vocab.get("grad", -1)
        return (token_ids[:, -1] == div_id) & (token_ids[:, -2] == grad_id)

    def div_grad_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a grad div`` -> ``… a lap``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-3]
        a = token_ids[:, -3:-2]

        lap_id = vocab.get("lap", -1)
        result = torch.cat([
            prefix,
            a,
            token_ids.new_full((B, 1), lap_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="div_grad_equals_lap",
            category=RuleCategory.VECTOR_CALCULUS,
            description="div(grad(a)) = lap(a)",
            pattern_length=3,
            output_length=2,
            match_fn=match_div_grad,
            transform_fn=div_grad_transform,
        )
    )

    # curl(grad) = 0
    def match_curl_grad(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        curl_id = vocab.get("curl", -1)
        grad_id = vocab.get("grad", -1)
        return (token_ids[:, -1] == curl_id) & (token_ids[:, -2] == grad_id)

    def curl_grad_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… a grad curl`` -> ``… 0``"""
        B, L = token_ids.shape
        prefix = token_ids[:, :-3]

        scalar_id = vocab.get("__scalar__", -1)
        result = torch.cat([
            prefix,
            token_ids.new_full((B, 1), scalar_id)
        ], dim=-1)
        return result

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="curl_grad_equals_zero",
            category=RuleCategory.VECTOR_CALCULUS,
            description="curl(grad(a)) = 0",
            pattern_length=3,
            output_length=1,
            match_fn=match_curl_grad,
            transform_fn=curl_grad_transform,
        )
    )

    # Cross product identities (if curl represents cross product with gradient)
    # curl(a*u) = a*curl(u) + grad(a) x u (but x not in RPN)
    # Too complex for suffix-only rewrite

    return rule_set