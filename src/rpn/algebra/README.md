# Simplified Modular Algebra Rules System

## Architecture

The modular algebra rules system provides a clean, extensible approach to
defining and applying algebraic equivalence rules for contrastive learning
with RPN embeddings.

### Core Components

1. **AlgebraicRule** - Base interface for all rules
2. **SimpleAlgebraicRule** - Concrete rule implementation
3. **AlgebraicRuleSet** - Container for related rules
4. **CompositeAlgebraicRuleSet** - Combines multiple rule sets

### Key Features

- **Modular Design**: Rules are organized into domain-specific sets (arithmetic, jacobian, etc.)
- **Tensor-based**: All rules operate on token ID tensors for neural network training
- **Extensible**: New rule domains can be easily added
- **Composable**: Rule sets can be combined for complex transformations

### Usage

```python
from qg.solver.opt.operator.rpn.algebra import create_composite_ruleset

# Create rule set with vocabulary
rules = create_composite_ruleset(TOKEN_TO_ID)

# Apply transformations
transformed_tokens = rules.apply_random_rule(token_ids)
```

### Rule Categories

The system supports these algebraic domains:

- **Arithmetic**: Commutativity, associativity, distributivity
- **Jacobian**: Operator properties, expansion rules
- **Calculus**: Derivative rules
- **Trigonometric**: Trigonometric identities
- **Vector Calculus**: Gradient, divergence, curl operations

### Integration with Contrastive Learning

The system is designed to work with:
1. RPN token ID tensors
2. Vocabulary-aware pattern matching
3. Batch processing for GPU efficiency
4. Rule composition for complex transformations

The modular design allows for:
- Easy extension with new rule sets
- Selective application of rules by category
- Independent testing of algebraic transformations
- Integration with neural network training pipelines