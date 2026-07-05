"""Aggregate all domain rule factories into one composite rule set."""

from typing import Dict, Optional

from .modular_rules import CompositeAlgebraicRuleSet
from .arithmetic_modular import create_arithmetic_rules
from .jacobian_modular import create_jacobian_rules
from .calculus_modular import create_calculus_rules
from .exponent_modular import create_exponent_rules
from .logarithm_modular import create_logarithm_rules
from .trigonometric_modular import create_trigonometric_rules
from .vector_calculus_modular import create_vector_calculus_rules


def create_composite_ruleset(
    vocab: Dict[str, int],
    pad_token_id: Optional[int] = None,
) -> CompositeAlgebraicRuleSet:
    """
    Single entry point: arithmetic + Jacobian rules that are implemented today,
    plus placeholder rule sets for domains to be filled in later.
    """
    return CompositeAlgebraicRuleSet(
        create_arithmetic_rules(vocab, pad_token_id),
        create_jacobian_rules(vocab, pad_token_id),
        create_calculus_rules(vocab, pad_token_id),
        create_exponent_rules(vocab, pad_token_id),
        create_logarithm_rules(vocab, pad_token_id),
        create_trigonometric_rules(vocab, pad_token_id),
        create_vector_calculus_rules(vocab, pad_token_id),
    )
