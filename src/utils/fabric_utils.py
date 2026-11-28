"""
Lightning Fabric utilities for device management, precision, and checkpointing.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger


def create_fabric(
    accelerator: str = "auto",
    precision: str = "32-true",
    devices: Union[int, str] = "auto",
    loggers: Optional[list] = None,
    **kwargs,
) -> Fabric:
    """
    Create and configure a Lightning Fabric instance.

    Args:
        accelerator: "auto", "cpu", "cuda", "mps", "tpu"
        precision: "32-true", "16-mixed", "bf16-mixed", "64-true"
        devices: Number of devices or "auto"
        loggers: List of Fabric loggers (CSVLogger, TensorBoardLogger, etc.)
        **kwargs: Additional Fabric arguments

    Returns:
        Configured Fabric instance (not yet launched)
    """
    if loggers is None:
        loggers = []

    fabric = Fabric(
        accelerator=accelerator,
        precision=precision,
        devices=devices,
        loggers=loggers,
        **kwargs,
    )
    return fabric


class FabricDeviceManager:
    """
    Manages device placement, buffer registration, and checkpointing via Fabric.

    Replaces manual .to(device) and register_buffer calls with Fabric-aware versions
    that work correctly across single/multi-GPU, DDP, and precision settings.
    """

    def __init__(self, fabric: Fabric):
        self.fabric = fabric
        self._device = fabric.device

    @property
    def device(self) -> torch.device:
        return self._device

    def to_device(self, module: nn.Module) -> nn.Module:
        """
        Move module to Fabric device. Handles DDP wrapping if needed.
        Call this once after model creation, before training.
        """
        return self.fabric.to_device(module)

    def register_buffer(
        self,
        module: nn.Module,
        name: str,
        tensor: torch.Tensor,
        persistent: bool = True,
    ) -> None:
        """
        Register a buffer on a module with correct device placement.

        Args:
            module: Target module
            name: Buffer name
            tensor: Tensor to register (will be moved to device)
            persistent: Whether buffer is part of state_dict
        """
        # Move tensor to correct device first
        tensor = self.fabric.to_device(tensor)
        module.register_buffer(name, tensor, persistent=persistent)

    def setup_optimizer(self, optimizer: optim.Optimizer) -> optim.Optimizer:
        """Wrap optimizer for Fabric (handles DDP, precision)."""
        return self.fabric.setup_optimizers(optimizer)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model, optimizer, and metadata checkpoint via Fabric.
        Handles DDP state_dict correctly.
        """
        state = {"model": model}
        if optimizer is not None:
            state["optimizer"] = optimizer
        if metadata is not None:
            state["metadata"] = metadata

        self.fabric.save(path, state)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint via Fabric. Returns metadata dict.
        """
        # First load the raw checkpoint to extract metadata (Fabric.load modifies in place)
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})

        state = {"model": model}
        if optimizer is not None:
            state["optimizer"] = optimizer

        self.fabric.load(path, state)
        return metadata

    def backward(self, loss: torch.Tensor) -> None:
        """Fabric-aware backward pass (handles precision scaling)."""
        self.fabric.backward(loss)

    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        max_norm: float = 1.0,
    ) -> None:
        """Fabric-aware gradient clipping."""
        self.fabric.clip_gradients(model, optimizer, max_norm=max_norm)