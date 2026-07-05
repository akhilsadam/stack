# RPN Test Suite

Comprehensive test suite for the RPN (Reverse Polish Notation) PDE compiler and learned token embeddings.

## Test Organization

Tests are organized by component:

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| `test_embeddings.py` | Token vocabulary and embedding tests | Vocabulary structure, token normalization, tokenization, embedding forward passes |
| `test_compiler.py` | RPN compiler interface tests | IR utilities, compiler initialization, expression building (assumes compiler is correct) |
| `test_algebra.py` | Algebraic rule mechanics tests | Rule structures, rule application, padding behavior, random positive view generation |
| `test_contrastive.py` | Contrastive learning tests | InfoNCE loss, masked pooling, trainer initialization, forward passes |
| `test_generator.py` | RPN generator tests | Vocabulary creation, expression generation, batch generation, complexity limits |
| `test_integration.py` | Integration tests | End-to-end workflows, component interaction, scalar parameter flow |

**Note:** Algebra rule tests verify mechanics (tensor shapes, padding, rule application) but **not mathematical correctness** of the rules themselves.

## Running Tests

### Quick Test Runner
```bash
export PYTHONPATH=packages/qg/src
python packages/qg/src/qg/solver/opt/operator/rpn/tests/run_all_tests.py
```

### Individual Test Modules
```bash
# Run all tests
python -m pytest packages/qg/src/qg/solver/opt/operator/rpn/tests/ -v

# Run specific module
python -m pytest packages/qg/src/qg/solver/opt/operator/rpn/tests/test_generator.py -v
python -m pytest packages/qg/src/qg/solver/opt/operator/rpn/tests/test_algebra.py -v
```

### Direct Import (for debugging)
```python
import sys
sys.path.insert(0, 'packages/qg/src')

from qg.solver.opt.operator.rpn.tests.test_generator import test_generator_initialization
test_generator_initialization()
```

## Test Coverage

### What's Tested
- ✅ **Vocabulary consistency**: All tokens have proper ID mappings and categories
- ✅ **Token normalization**: Aliases and case normalization work correctly
- ✅ **Batch tokenization**: RPN strings convert to token ID tensors with scalar values
- ✅ **Embedding forward passes**: Embedder produces correct shapes, no NaN/Inf
- ✅ **Algebra rule mechanics**: Rule application preserves batch dimensions, handles padding
- ✅ **Contrastive training**: InfoNCE loss computation, gradient flow
- ✅ **RPN generation**: Expressions respect complexity limits, have valid tokens
- ✅ **Integration**: Components work together in realistic workflows

### What's NOT Tested (by design)
- ❌ **Algebra rule correctness**: Tests assume rules are mathematically correct
- ❌ **Compiler correctness**: Tests assume compiler produces correct PDE components
- ❌ **Learning convergence**: No tests for whether embeddings actually learn algebra
- ❌ **Numerical precision**: Tests don't verify floating-point accuracy

## Testing Philosophy

Tests follow these principles:

1. **Independent**: Each test module focuses on one component
2. **Comprehensive**: Cover edge cases, error conditions, different batch sizes
3. **Non-destructive**: Don't modify files or state
4. **Fast**: Use small tensors, mock complex dependencies
5. **Clear failures**: Tests have descriptive error messages

## Adding New Tests

When adding new functionality:

1. Add tests to existing module if functionality fits
2. Create new test module if testing new component
3. Follow existing patterns for test naming and structure
4. Include edge cases and error conditions
5. Run `run_all_tests.py` to ensure no regressions

Example test template:
```python
def test_new_feature():
    """Test description."""
    # Setup
    # Exercise
    # Verify
    # Cleanup (if needed)
```

## Test Dependencies

Tests assume:
- `torch` is available
- Vocabulary from `embeddings.py` is stable
- Python imports work (may require `PYTHONPATH` setup)
- No GPU required (tests work on CPU)

## Common Issues

### Import Errors
```bash
# Set PYTHONPATH
export PYTHONPATH=packages/qg/src
```

### Pytest Collection Errors
```bash
# Use direct runner instead
python packages/qg/src/qg/solver/opt/operator/rpn/tests/run_all_tests.py
```

### Missing Dependencies
```bash
# Install required packages
pip install torch pytest numpy
```

## Test Results

See [TEST_RESULTS_SUMMARY.md](TEST_RESULTS_SUMMARY.md) for detailed results.