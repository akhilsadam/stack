# Design: Modern ML Reproducibility & GPU Support for Stack Autoencoder

## Goal
Add comprehensive reproducibility (seeding, config tracking, experiment management) and GPU support (device-aware buffers, deterministic algorithms, mixed precision) to the stack autoencoder codebase, with all outputs managed in `_output/` with git tracking and hashing.

## Architecture Overview

### 1. Seeding Utility (`src/utils/reproducibility.py`)
Single `set_seed(seed: int, deterministic: bool = True)` function covering:
- `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `numpy.random.seed`, `random.seed`
- `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `PYTHONHASHSEED` env var
- Returns a `ReproducibilityConfig` dataclass for logging

### 2. Device Management via Lightning Fabric (`src/utils/fabric_utils.py`)
Lightweight wrapper around `lightning.Fabric`:
- `create_fabric(accelerator="auto", precision="32-true", **kwargs)` → returns configured Fabric
- `FabricDeviceManager` class: `.to_device(module)`, `.register_buffer(module, name, tensor)`, `.save_checkpoint()`, `.load_checkpoint()`
- Handles device placement, DDP-ready patterns, precision (FP32/BF16/FP16)
- Replaces manual `.to(device)` calls with fabric-managed lifecycle

### 3. Run Manager (`src/utils/run_manager.py`)
`RunManager` class using Fabric's loggers + stdlib:
- Creates `_output/{app_name}/{timestamp}_{git_short_hash}/` directory
- Saves: `config.yaml` (resolved config), `git_metadata.json` (commit, diff, branch, dirty), `config_hash.txt` (SHA256)
- Artifact tracking: `save_artifact(name, obj, step=None)` → saves with SHA256 manifest
- Integrates with Fabric's `TensorBoardLogger` / `CSVLogger` for metrics
- `log_metrics(dict, step)`, `log_hyperparams(dict)`

### 4. Integration Points
- `run_search.py`: Initialize Fabric + RunManager at start; wrap model/data; use Fabric for training loop
- App `_eval.py`: Accept Fabric/RunManager for device-aware tensor creation
- Stack/Operator modules: Use `fabric.to_device()` or registration helpers for buffers

## Trade-offs Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Custom RunManager (stdlib only) | Zero deps, full control | Re-invent logging, no ecosystem | Not chosen |
| Lightning Fabric (chosen) | Device/precision/logging built-in, DDP-ready, minimal overhead | Requires `lightning` install | Chosen |
| Full Lightning Trainer | Maximum features | Forces restructuring, high overhead | Too invasive |
| wandb/mlflow | Cloud tracking, UI | External deps, auth, network | Overkill for now |

## File Structure Changes

```
src/
├── utils/
│   ├── __init__.py
│   ├── reproducibility.py      # NEW: set_seed(), ReproducibilityConfig
│   ├── fabric_utils.py         # NEW: FabricDeviceManager, create_fabric()
│   └── run_manager.py          # NEW: RunManager class
├── arch/
│   ├── layer.py                # MODIFY: register_buffer → fabric registration
│   ├── cope.py                 # MODIFY: register_buffer → fabric registration
│   └── stack_rope.py           # MODIFY: register_buffer → fabric registration
├── space/
│   ├── tokenizer.py            # MODIFY: Stack.stack buffer registration
│   └── operator.py             # CHECK: buffer usage
└── search/
    └── run_search.py           # MODIFY: Fabric + RunManager integration
apps/
└── */_eval.py                  # MODIFY: device-aware tensor creation
```

## Success Criteria
1. `python -m search.run_search` runs on GPU with `accelerator="cuda"` and produces identical results to CPU for same seed
2. Each run creates `_output/qg_apriori_deep/2026-11-27_14-30-00_abc123def/` with config, git metadata, metrics
3. `torch.use_deterministic_algorithms(True)` passes without errors
5. Mixed precision (`precision="16-mixed"`) works as optional flag
6. Checkpoint save/load via Fabric works across devices