# tests/test_fabric_utils.py
import torch
import torch.nn as nn
import warnings
from src.utils.fabric_utils import create_fabric, FabricDeviceManager

# Suppress pynvml deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.register_buffer("const_buffer", torch.ones(5))

    def forward(self, x):
        return self.linear(x) + self.const_buffer


def test_create_fabric_cpu():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    assert fabric is not None
    assert fabric.device.type == "cpu"


def test_fabric_device_manager_to_device():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()
    manager = FabricDeviceManager(fabric)
    model = TestModule()

    model = manager.to_device(model)
    assert next(model.parameters()).device.type == "cpu"
    assert model.const_buffer.device.type == "cpu"


def test_fabric_register_buffer():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()
    manager = FabricDeviceManager(fabric)
    model = TestModule()
    model = manager.to_device(model)

    # Register new buffer through manager
    new_buffer = torch.zeros(3)
    manager.register_buffer(model, "new_buffer", new_buffer)
    assert hasattr(model, "new_buffer")
    assert model.new_buffer.device.type == "cpu"
    assert model.new_buffer.shape == (3,)


def test_fabric_checkpoint_save_load(tmp_path):
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()
    manager = FabricDeviceManager(fabric)
    model = TestModule()
    model = manager.to_device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Save
    ckpt_path = tmp_path / "test_ckpt.pt"
    manager.save_checkpoint(ckpt_path, model, optimizer, {"step": 42})

    # Load into new model
    model2 = TestModule()
    model2 = manager.to_device(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    state = manager.load_checkpoint(ckpt_path, model2, optimizer2)

    assert state["step"] == 42
    assert torch.allclose(model.linear.weight, model2.linear.weight)