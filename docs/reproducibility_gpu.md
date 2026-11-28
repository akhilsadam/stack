# Reproducibility & GPU Support Guide

## Quick Start

### CPU (deterministic)
```bash
python -m search.run_search
```

### GPU (auto-detect)
```bash
python -m search.run_search
# Or explicitly:
ACCELERATOR=cuda python -m search.run_search
```

### Mixed Precision (FP16)
```bash
PRECISION=16-mixed python -m search.run_search
```

### BF16 (Ampere+ GPUs)
```bash
PRECISION=bf16-mixed python -m search.run_search
```

## Configuration

Add to your app's `config.yaml`:

```yaml
seed: 42
deterministic: true       # Enable torch deterministic algorithms
accelerator: "auto"       # auto, cpu, cuda, mps
precision: "32-true"      # 32-true, 16-mixed, bf16-mixed, 64-true
```

## Output Structure

Each run creates:

```
_output/
└── qg_apriori_deep/
    └── 2026-11-27_14-30-00_abc123def/
        ├── config.yaml           # Resolved configuration
        ├── git_metadata.json     # Commit, branch, diff, dirty status
        ├── config_hash.txt       # SHA256 of config for change detection
        ├── metrics/              # CSV logs from CSVLogger
        │   └── metrics.csv
        ├── tb/                   # TensorBoard logs
        │   └── events.out.tfevents...
        └── artifacts/
            ├── manifest.json     # Artifact index with SHA256 hashes
            ├── final_model.pt    # Model checkpoint
            └── y_hat_*.png       # Training plots
```

## Reproducibility

- Same seed + `deterministic=True` → bitwise identical results (CPU)
- GPU determinism requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set automatically)
- Config hash allows detecting parameter changes between runs
- Git metadata tracks exact code state

## Checkpointing

```python
# Save
manager.save_checkpoint("checkpoint.pt", model, optimizer, {"step": 1000})

# Load
state = manager.load_checkpoint("checkpoint.pt", model, optimizer)
print(state["step"])  # 1000
```

## Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 -m search.run_search
```

Fabric handles DDP automatically - no code changes needed.

## API Reference

### FabricDeviceManager

```python
from src.utils.fabric_utils import create_fabric, FabricDeviceManager

fabric = create_fabric(accelerator="auto", precision="32-true")
fabric.launch()

manager = FabricDeviceManager(fabric)

# Move model to device
model = manager.to_device(model)

# Register buffers (automatically placed on correct device)
manager.register_buffer(model, "my_buffer", torch.zeros(10))

# Setup optimizer (handles precision scaling, DDP)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = manager.setup_optimizer(optimizer)

# Training step
loss = compute_loss()
manager.backward(loss)  # Handles mixed precision
manager.clip_gradients(model, optimizer, max_norm=1.0)
optimizer.step()

# Checkpointing
manager.save_checkpoint("ckpt.pt", model, optimizer, {"step": 100})
state = manager.load_checkpoint("ckpt.pt", model, optimizer)
```

### RunManager

`RunManager` is a context manager. Inside the block it patches the global
entry points the training loop touches, so no plumbing is needed in the
training code itself:

```python
from src.utils.run_manager import RunManager

run_manager = RunManager(
    fabric=fabric,
    app_name="my_experiment",
    config={"lr": 1e-3, "batch_size": 32},
)
run_manager.log_hyperparams({"lr": 1e-3, "model": "transformer"})

with run_manager:
    tokenizer._train(evaluator, **cfg)   # no fabric / manager / run_manager args

# Outside the block you can still log explicitly
run_manager.log_metrics({"loss": 0.5, "acc": 0.9}, step=10)
run_manager.save_artifact("model", model, step=100)
run_manager.save_artifact("plot", fig, step=50)
run_manager.save_artifact("results", {"metrics": {...}}, step=100)
run_manager.finalize()
```

Inside `with run_manager:` the following are patched automatically:

| What | Effect |
|------|--------|
| `plt.savefig("foo.png")` | Writes to `<run_dir>/artifacts/foo.png` (original filename kept) and records it in the manifest |
| `logger.info(msg, extra={"metrics": {...}, "step": n})` | Metrics are appended to `metrics/metrics.csv` |
| `print(...)` / `sys.stdout` | Tee'd to the console and captured to `artifacts/stdout.txt` |
| `sys.stderr` | Captured to `artifacts/stderr.txt` |

So evaluators keep calling `plt.savefig("y_hat_{iter}.png")` as before — only
the destination changes.

### Stack Tokenizer Integration

`_train` takes only the evaluator and its own hyperparameters. Device
placement, metric logging, and artifact capture are all ambient:

```python
tokenizer._train(evaluator, **cfg)
```

To emit metrics from your own code, use the standard logging `extra` convention:

```python
logger.info("", extra={"metrics": {"loss": 0.5}, "step": 10})
```

### Evaluator Device Awareness

Evaluators should accept a `device` parameter for proper tensor placement:

```python
evaluator = Evaluator(device=fabric.device, **cfg)
evaluator = manager.to_device(evaluator)
```

## Output Layout

```
_output/{app_name}/{YYYY-MM-DD_HH-MM-SS}_{git_short_hash}/
├── config.yaml            # resolved config
├── hyperparameters.yaml
├── git_metadata.json      # commit, branch, dirty, diff, remote
├── git_diff.patch         # full git diff, human readable
├── app_hash.txt           # SHA256 of the whole app directory
├── app_file_hashes.txt    # per-file SHA256
├── metrics/metrics.csv
└── artifacts/
    ├── manifest.json      # SHA256 per artifact
    ├── stdout.txt
    ├── stderr.txt
    ├── y_hat_0.png        # plots, original filenames
    └── tokenizer_final.pt
```

## Implementation Summary

This implementation provides:

1. **Lightning Fabric** for device/precision management (minimal overhead, DDP-ready)
2. **RunManager** for git-tracked experiment directories with:
   - Timestamp + git hash directory naming
   - Config YAML + app-directory SHA256
   - Git metadata (commit, branch, diff, dirty status) + readable `git_diff.patch`
   - CSV/TensorBoard metric logging
   - Artifact manifest with SHA256 hashes
   - Context-manager patching of `plt.savefig`, logging metrics, stdout/stderr
3. **FabricDeviceManager** for:
   - Correct device placement
   - Buffer registration
   - Precision-aware backward/clipping
   - Checkpoint save/load
4. **Deterministic algorithms** support via config
5. **Integration** into existing custom training loop (MCTS + gradient descent)

See `docs/superpowers/specs/2026-11-27-repro-gpu-design.md` for architecture details.
