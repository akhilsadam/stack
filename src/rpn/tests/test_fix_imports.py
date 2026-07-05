"""
Quick test to check imports.
"""

import sys
sys.path.insert(0, 'packages/qg/src')

try:
    from qg.solver.opt.operator.rpn.contrastive import ContrastiveRPN
    print("✓ ContrastiveRPN imported")
except ImportError as e:
    print(f"✗ ContrastiveRPN: {e}")

# List what's available
from qg.solver.opt.operator.rpn.contrastive import *
print("\nAvailable in contrastive:")
import inspect
for name in dir():
    if not name.startswith('_'):
        print(f"  {name}")

# Check batch_tokenize_rpn signature
import inspect
from qg.solver.opt.operator.rpn.embeddings import batch_tokenize_rpn
sig = inspect.signature(batch_tokenize_rpn)
print(f"\nbatch_tokenize_rpn signature: {sig}")