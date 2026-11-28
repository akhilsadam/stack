"""
Run Manager for experiment tracking with git integration.
Creates timestamped output directories with config, git metadata, and artifact tracking.
"""

import hashlib
import io
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger

# Optional tensorboard
try:
    from lightning.fabric.loggers import TensorBoardLogger
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TensorBoardLogger = None
    TENSORBOARD_AVAILABLE = False

# Suppress pynvml deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


class RunManager:
    """
    Manages experiment runs: output directories, config/git metadata, metrics logging, artifacts.

    Creates directories: _output/{app_name}/{YYYY-MM-DD_HH-MM-SS}_{git_short_hash}/
    Saves: config.yaml, git_metadata.json, config_hash.txt
    Logs: metrics via Fabric loggers (CSV, TensorBoard)
    Artifacts: model checkpoints, plots, with SHA256 manifest
    """

    def __init__(
        self,
        fabric: Fabric,
        app_name: str,
        config: Dict[str, Any],
        output_base: Optional[Union[str, Path]] = None,
    ):
        self.fabric = fabric
        self.app_name = app_name
        self.config = config

        # Project root is fixed (parent of src/ directory)
        self._project_root = Path(__file__).parent.parent.parent

        # Resolve output base
        if output_base is None:
            output_base = self._project_root / "_output"
        self.output_base = Path(output_base)

        # Create run directory with timestamp and git hash
        self.run_dir = self._create_run_dir()

        # Save config and git metadata
        self._save_config()
        self._save_git_metadata()
        self._save_app_hash()

        # Setup Fabric loggers pointing to run_dir
        self._setup_loggers()

        # Artifact manifest
        self._manifest: Dict[str, Dict[str, Any]] = {}
        self._manifest_path = self.run_dir / "artifacts" / "manifest.json"
        self._artifacts_dir = self.run_dir / "artifacts"
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_run_dir(self) -> Path:
        """Create timestamped run directory with git short hash."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        git_hash = self._get_git_short_hash()
        dir_name = f"{timestamp}_{git_hash}"
        run_dir = self.output_base / self.app_name / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_git_short_hash(self) -> str:
        """Get short git commit hash, or 'nogit' if not available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self._project_root,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "nogit"

    def _get_git_metadata(self) -> Dict[str, Any]:
        """Collect git metadata for reproducibility."""
        meta = {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": False,
            "diff": "",
            "remote_url": "unknown",
        }
        try:
            repo_root = self._project_root

            # Commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if result.returncode == 0:
                meta["commit"] = result.stdout.strip()

            # Branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if result.returncode == 0:
                meta["branch"] = result.stdout.strip()

            # Dirty status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if result.returncode == 0:
                meta["dirty"] = len(result.stdout.strip()) > 0

            # Diff (staged + unstaged)
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if result.returncode == 0:
                meta["diff"] = result.stdout.strip()[:10000]  # Limit size

            # Remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if result.returncode == 0:
                meta["remote_url"] = result.stdout.strip()

        except Exception:
            pass
        return meta

    def _save_config(self) -> None:
        """Save resolved config to config.yaml."""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def _save_git_metadata(self) -> None:
        """Save git metadata to git_metadata.json and git diff to separate file."""
        git_meta = self._get_git_metadata()
        git_path = self.run_dir / "git_metadata.json"
        with open(git_path, "w") as f:
            json.dump(git_meta, f, indent=2)

        # Save git diff to separate readable file
        diff_path = self.run_dir / "git_diff.patch"
        diff_content = git_meta.get("diff", "")
        diff_path.write_text(diff_content)

    def _save_app_hash(self) -> None:
        """Save SHA256 hash of the full app directory for reproducibility."""
        app_dir = self._project_root / "src" / "apps" / self.app_name

        # Collect all Python files in the app directory
        file_hashes = []
        for py_file in sorted(app_dir.rglob("*.py")):
            if py_file.is_file():
                with open(py_file, "rb") as f:
                    content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes.append((str(py_file.relative_to(app_dir)), file_hash))

        # Create combined hash from all file hashes
        combined = "".join(f"{name}:{h}\n" for name, h in file_hashes)
        app_hash = hashlib.sha256(combined.encode()).hexdigest()

        # Save combined hash
        hash_path = self.run_dir / "app_hash.txt"
        hash_path.write_text(app_hash)

        # Also save per-file hashes for detailed inspection
        detail_path = self.run_dir / "app_file_hashes.txt"
        with open(detail_path, "w") as f:
            for name, h in file_hashes:
                f.write(f"{name}:{h}\n")

    def _setup_loggers(self) -> None:
        """Configure Fabric loggers to write to run directory."""
        # CSV logger for metrics (always available)
        csv_logger = CSVLogger(root_dir=self.run_dir, name="metrics", version="")

        # TensorBoard logger (optional) - wrap in try/except
        tb_logger = None
        try:
            from lightning.fabric.loggers import TensorBoardLogger
            tb_logger = TensorBoardLogger(root_dir=self.run_dir, name="tb", version="")
        except (ImportError, ModuleNotFoundError):
            tb_logger = None

        # Set the loggers on the fabric (must be done before or at launch)
        loggers = [csv_logger]
        if tb_logger is not None:
            loggers.append(tb_logger)
        self.fabric._loggers = loggers

        # Store for reference
        self._csv_logger = csv_logger
        self._tb_logger = tb_logger

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics via CSV logger directly."""
        # Log via CSV logger directly (works regardless of Fabric logger setup)
        self._csv_logger.log_metrics(metrics, step=step)
        self._csv_logger.save()

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # CSVLogger doesn't support hyperparameters, log to YAML file instead
        hp_path = self.run_dir / "hyperparameters.yaml"
        with open(hp_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        # Also log via fabric if loggers are set up (for TensorBoard)
        try:
            self.fabric.log_hyperparams(params)
        except Exception:
            pass

    def save_artifact(
        self,
        name: str,
        obj: Any,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save an artifact (model, plot, data) with SHA256 manifest entry.

        Args:
            name: Artifact name (e.g., "model", "plot_loss")
            obj: Object to save (torch module, dict, figure, etc.)
            step: Training step/iteration
            metadata: Additional metadata

        Returns:
            Path to saved artifact
        """
        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension and save
        if step is not None:
            filename = f"{name}_step_{step}"
        else:
            filename = name

        # Handle different object types
        if hasattr(obj, "state_dict"):  # torch.nn.Module
            filepath = artifacts_dir / f"{filename}.pt"
            import torch
            torch.save(obj.state_dict(), filepath)
        elif hasattr(obj, "savefig"):  # matplotlib figure
            filepath = artifacts_dir / f"{filename}.png"
            obj.savefig(filepath, dpi=300)
        elif isinstance(obj, dict):
            filepath = artifacts_dir / f"{filename}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(obj, f)
        elif isinstance(obj, str):
            # Save strings as plain text
            filepath = artifacts_dir / f"{filename}.txt"
            with open(filepath, "w") as f:
                f.write(obj)
        else:
            # Fallback: pickle
            import pickle
            filepath = artifacts_dir / f"{filename}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)

        # Compute hash
        with open(filepath, "rb") as f:
            content = f.read()
        file_hash = hashlib.sha256(content).hexdigest()

        # Update manifest
        self._manifest[name] = {
            "file": filepath.name,
            "hash": file_hash,
            "step": step,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._save_manifest()

        return filepath

    def _save_manifest(self) -> None:
        """Save artifact manifest."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def finalize(self) -> None:
        """Finalize run: save final manifest, close loggers."""
        self._save_manifest()
        # Fabric handles logger cleanup on teardown

    # ── context manager: patch global entry points ────────────────
    #
    # Everything the training loop touches globally is patched here, so
    # ``Stack._train`` needs no knowledge of Fabric, the RunManager, or
    # logging plumbing.  Inside ``with run_manager:``:
    #
    #   * ``plt.savefig("foo.png")``  -> ``<run_dir>/artifacts/foo.png``
    #   * ``logger.info(..., extra={"metrics": {...}})`` -> metrics CSV
    #   * stdout/stderr -> ``stdout.txt`` / ``stderr.txt`` artifacts

    def __enter__(self) -> "RunManager":
        from matplotlib import pyplot as plt

        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        # ── matplotlib: redirect saves into the artifacts dir ──
        self._orig_savefig = plt.savefig
        plt.savefig = self._patched_savefig

        # ── logging: capture stdout/stderr text + pick up metrics ──
        self._stdout_capture = io.StringIO()
        self._stderr_capture = io.StringIO()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._stdout_capture)
        sys.stderr = _Tee(self._orig_stderr, self._stderr_capture)

        self._log_handler = logging.StreamHandler(self._stdout_capture)
        self._log_handler.setLevel(logging.INFO)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(self._log_handler)

        self._metrics_handler = _MetricsHandler(self)
        logging.getLogger().addHandler(self._metrics_handler)

        # Make sure INFO records reach the handlers regardless of what the
        # emitting logger's own level happens to be.
        root = logging.getLogger()
        self._orig_root_level = root.level
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        from matplotlib import pyplot as plt

        plt.savefig = self._orig_savefig

        logging.getLogger().removeHandler(self._log_handler)
        logging.getLogger().removeHandler(self._metrics_handler)

        logging.getLogger().setLevel(self._orig_root_level)

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        self._save_stream("stdout", self._stdout_capture.getvalue())
        self._save_stream("stderr", self._stderr_capture.getvalue())

    def _patched_savefig(self, fname, *args, **kwargs):
        """matplotlib ``savefig`` -> write into the artifacts dir instead."""
        name = Path(fname).name
        target = self._artifacts_dir / name
        self._orig_savefig(target, *args, **kwargs)
        self._record_artifact(name, target)

    def _record_artifact(self, name: str, filepath: Path) -> None:
        """Add an externally-written file to the manifest."""
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        self._manifest[name] = {
            "file": filepath.name,
            "hash": file_hash,
            "step": None,
            "timestamp": time.time(),
            "metadata": {},
        }
        self._save_manifest()

    def _save_stream(self, name: str, text: str) -> None:
        if text:
            self.save_artifact(name, text, metadata={"stream": name})


class _Tee:
    """Write to two streams at once (console + capture buffer)."""

    def __init__(self, primary, capture):
        self.primary = primary
        self.capture = capture

    def write(self, text):
        self.primary.write(text)
        self.primary.flush()
        self.capture.write(text)

    def flush(self):
        self.primary.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()


class _MetricsHandler(logging.Handler):
    """Pick up ``extra={"metrics": {...}}`` log records and log them."""

    def __init__(self, run_manager: RunManager):
        super().__init__(level=logging.INFO)
        self.run_manager = run_manager

    def emit(self, record: logging.LogRecord) -> None:
        metrics = getattr(record, "metrics", None)
        if metrics is None:
            return
        self.run_manager.log_metrics(metrics, step=getattr(record, "step", None))
