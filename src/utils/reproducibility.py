"""
Reproducibility utilities for deterministic ML experiments.
"""

import os
import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Configuration snapshot for reproducibility tracking."""
    seed: int
    deterministic: bool
    torch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    python_hash_seed: str
    cublas_workspace_config: Optional[str]


def set_seed(seed: int, deterministic: bool = True) -> ReproducibilityConfig:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed integer
        deterministic: If True, enable torch deterministic algorithms (slower but reproducible)

    Returns:
        ReproducibilityConfig with all settings for logging
    """
    # Python hash seed must be set before any module imports that use hash
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Core seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    cublas_config = None
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cublas_config = ":4096:8"
        torch.use_deterministic_algorithms(True)
        # Ensure cuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return ReproducibilityConfig(
        seed=seed,
        deterministic=deterministic,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        python_hash_seed=str(seed),
        cublas_workspace_config=cublas_config,
    )