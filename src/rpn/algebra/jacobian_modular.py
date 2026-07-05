"""
Jacobian equivalence rules on RPN token ID tensors.
"""

import torch
from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet, SimpleAlgebraicRule, RuleCategory


def create_jacobian_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    """Antisymmetry: ``J(f,g) = -J(g,f)``  <=>  ``f g jacobian`` ~ ``g f jacobian neg``."""
    rule_set = AlgebraicRuleSet("jacobian", vocab, pad_token_id=pad_token_id)

    def jacobian_match(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        B, L = token_ids.shape
        if L < 3:
            return torch.zeros(B, dtype=torch.bool, device=token_ids.device)

        j_id = vocab.get("jacobian", -1)
        return token_ids[:, -1] == j_id

    def jacobian_antisymmetry_transform(token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        """``… f g jacobian``  ->  ``… g f jacobian neg``"""
        B, L = token_ids.shape
        neg_id = vocab["neg"]
        prefix = token_ids[:, :-3]
        f = token_ids[:, -3:-2]
        g = token_ids[:, -2:-1]
        jac = token_ids[:, -1:]
        neg = torch.full((B, 1), neg_id, dtype=token_ids.dtype, device=token_ids.device)
        new_suffix = torch.cat([g, f, jac, neg], dim=-1)
        return torch.cat([prefix, new_suffix], dim=-1)

    rule_set.add_rule(
        SimpleAlgebraicRule(
            name="jacobian_antisymmetry",
            category=RuleCategory.JACOBIAN,
            description="J(f,g) = -J(g,f)",
            pattern_length=3,
            output_length=4,
            match_fn=jacobian_match,
            transform_fn=jacobian_antisymmetry_transform,
        )
    )

    return rule_set
