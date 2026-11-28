# tests/test_reproducibility.py
import torch
import numpy as np
import random
import os
from src.utils.reproducibility import set_seed, ReproducibilityConfig

def test_set_seed_basic():
    cfg = set_seed(42, deterministic=False)
    assert isinstance(cfg, ReproducibilityConfig)
    assert cfg.seed == 42
    assert cfg.deterministic is False

def test_set_seed_deterministic():
    cfg = set_seed(123, deterministic=True)
    assert cfg.deterministic is True
    # Verify env var set
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("PYTHONHASHSEED") == "123"

def test_reproducibility_torch():
    set_seed(42, deterministic=True)
    a = torch.randn(3, 3)
    set_seed(42, deterministic=True)
    b = torch.randn(3, 3)
    assert torch.allclose(a, b)

def test_reproducibility_numpy():
    set_seed(42, deterministic=True)
    a = np.random.randn(3, 3)
    set_seed(42, deterministic=True)
    b = np.random.randn(3, 3)
    assert np.allclose(a, b)

def test_reproducibility_random():
    set_seed(42, deterministic=True)
    a = [random.random() for _ in range(10)]
    set_seed(42, deterministic=True)
    b = [random.random() for _ in range(10)]
    assert a == b