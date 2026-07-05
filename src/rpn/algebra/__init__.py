"""
Algebraic equivalence rules for RPN token tensors (contrastive positives).

Import :func:`create_composite_ruleset` from :mod:`composite_modular` as the
main entry point; core types live in :mod:`modular_rules`.
"""

from .modular_rules import (
    AlgebraicRuleSet,
    CompositeAlgebraicRuleSet,
    RuleCategory,
    SimpleAlgebraicRule,
    TransformResult,
)
from .composite_modular import create_composite_ruleset
from .arithmetic_modular import create_arithmetic_rules
from .jacobian_modular import create_jacobian_rules
from .calculus_modular import create_calculus_rules
from .exponent_modular import create_exponent_rules
from .logarithm_modular import create_logarithm_rules
from .trigonometric_modular import create_trigonometric_rules
from .vector_calculus_modular import create_vector_calculus_rules

__all__ = [
    "AlgebraicRuleSet",
    "CompositeAlgebraicRuleSet",
    "RuleCategory",
    "SimpleAlgebraicRule",
    "TransformResult",
    "create_composite_ruleset",
    "create_arithmetic_rules",
    "create_jacobian_rules",
    "create_calculus_rules",
    "create_exponent_rules",
    "create_logarithm_rules",
    "create_trigonometric_rules",
    "create_vector_calculus_rules",
]
