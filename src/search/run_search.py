from __future__ import annotations

import os
import sys
import torch
import yaml
from tqdm import tqdm

# Add package root to path
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from space.tokenizer import Stack, Operator
from utils.fabric_utils import create_fabric, FabricDeviceManager
from utils.run_manager import RunManager

# App to run
app = "qg_apriori_deep"


import importlib.util
eval_path = os.path.join(_PKG, f"apps/{app}/_eval.py")
spec = importlib.util.spec_from_file_location("eval", eval_path)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

build_vocab = eval_module.build_vocab
Evaluator   = eval_module.Evaluator
_CONFIG     = os.path.join(_PKG, f"apps/{app}/config.yaml")


def main():
    # ── load configs ───────────────────────────────────────────────
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # ── Fabric setup ───────────────────────────────────────────────
    accelerator = cfg.get("accelerator", "auto")
    precision = cfg.get("precision", "32-true")
    devices = cfg.get("devices", "auto")

    fabric = create_fabric(accelerator=accelerator, precision=precision, devices=devices)
    fabric.launch()
    manager = FabricDeviceManager(fabric)

    device = fabric.device
    print(f"Device: {device}")
    print(f"Package root: {_PKG}")

    # ── RunManager for experiment tracking ─────────────────────────
    run_manager = RunManager(
        fabric=fabric,
        app_name=app,
        config=cfg,
    )

    # Log hyperparameters
    run_manager.log_hyperparams(cfg)

    # ── build components ───────────────────────────────────────────
    print("Building vocabulary and QG evaluator …")
    vocab = build_vocab(_Operator=Operator, **cfg)

    tokenizer = Stack(
        init_str = cfg.get('init'),
        vocab=vocab,
        **cfg,
    )

    # Move tokenizer to device using Fabric
    tokenizer = manager.to_device(tokenizer)

    # Create evaluator with device (buffers registered internally)
    evaluator = Evaluator(device=device, **cfg)
    # Move to device via Fabric (handles buffers automatically since they're registered)
    evaluator = manager.to_device(evaluator)

    # RunManager's context manager patches plt.savefig, logging metrics and
    # stdout/stderr — the training loop needs no knowledge of any of it.
    with run_manager:
        tokenizer._train(evaluator, **cfg)

    # Save final checkpoint
    run_manager.save_artifact("tokenizer_final", tokenizer)
    run_manager.finalize()

    print(f"\nRun complete. Output saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()