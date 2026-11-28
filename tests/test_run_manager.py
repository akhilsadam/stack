# tests/test_run_manager.py
import tempfile
import yaml
import json
import warnings
from pathlib import Path
from src.utils.run_manager import RunManager
from src.utils.fabric_utils import create_fabric

# Suppress pynvml deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


def test_run_manager_creates_output_dir():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    config = {"seed": 42, "lr": 0.01, "model": "test"}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override _output base
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config=config,
            output_base=Path(tmpdir),
        )

        run_dir = run_manager.run_dir
        assert run_dir.exists()
        assert "test_app" in str(run_dir)
        # Check timestamp_githash format
        parts = run_dir.name.split("_")
        assert len(parts) >= 2  # timestamp_hash

    # fabric.teardown() doesn't exist in this version


def test_run_manager_saves_config_and_git_metadata():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    config = {"seed": 42, "lr": 0.01, "nested": {"a": 1}}
    with tempfile.TemporaryDirectory() as tmpdir:
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config=config,
            output_base=Path(tmpdir),
        )

        run_dir = run_manager.run_dir

        # Check config.yaml saved
        config_path = run_dir / "config.yaml"
        assert config_path.exists()
        with open(config_path) as f:
            saved_config = yaml.safe_load(f)
        assert saved_config == config

        # Check git_metadata.json saved
        git_path = run_dir / "git_metadata.json"
        assert git_path.exists()
        with open(git_path) as f:
            git_meta = json.load(f)
        assert "commit" in git_meta
        assert "branch" in git_meta
        assert "dirty" in git_meta
        assert "diff" in git_meta

        # Check app_hash.txt
        hash_path = run_dir / "app_hash.txt"
        assert hash_path.exists()
        hash_content = hash_path.read_text().strip()
        assert len(hash_content) == 64  # SHA256 hex

        # Check app_file_hashes.txt also exists
        detail_path = run_dir / "app_file_hashes.txt"
        assert detail_path.exists()

        # Check git_diff.patch exists
        diff_path = run_dir / "git_diff.patch"
        assert diff_path.exists()

    # fabric.teardown() doesn't exist in this version


def test_run_manager_log_metrics():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    config = {"seed": 42}
    with tempfile.TemporaryDirectory() as tmpdir:
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config=config,
            output_base=Path(tmpdir),
        )

        # Log some metrics
        run_manager.log_metrics({"loss": 0.5, "acc": 0.9}, step=10)
        run_manager.log_metrics({"loss": 0.3, "acc": 0.95}, step=20)

        # Check CSV logger output
        csv_files = list(run_manager.run_dir.glob("**/*.csv"))
        assert len(csv_files) > 0

    # fabric.teardown() doesn't exist in this version


def test_run_manager_save_artifact():
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    config = {"seed": 42}
    with tempfile.TemporaryDirectory() as tmpdir:
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config=config,
            output_base=Path(tmpdir),
        )

        import torch
        model = torch.nn.Linear(10, 5)
        run_manager.save_artifact("model_step_100", model, step=100)

        # Check artifact saved with manifest
        artifacts = list(run_manager.run_dir.glob("artifacts/**/*.pt"))
        assert len(artifacts) == 1

        manifest_files = list(run_manager.run_dir.glob("artifacts/manifest*.json"))
        assert len(manifest_files) == 1

    # fabric.teardown() doesn't exist in this version


def test_run_manager_context_manager_redirects_savefig():
    """Inside the context, plt.savefig writes into the artifacts dir."""
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config={"seed": 42},
            output_base=Path(tmpdir),
        )

        from matplotlib import pyplot as plt

        with run_manager:
            fig = plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.savefig("some_plot.png")
            plt.close(fig)

        # Plot landed in artifacts, not the cwd
        assert (run_manager.run_dir / "artifacts" / "some_plot.png").exists()
        # And is tracked in the manifest
        with open(run_manager.run_dir / "artifacts" / "manifest.json") as f:
            manifest = json.load(f)
        assert "some_plot.png" in manifest

    # fabric.teardown() doesn't exist in this version


def test_run_manager_context_manager_logs_metrics():
    """logger.info(..., extra={'metrics': ...}) inside the context reaches the CSV."""
    import logging

    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config={"seed": 42},
            output_base=Path(tmpdir),
        )

        logger = logging.getLogger("test_metrics")
        with run_manager:
            logger.info("", extra={"metrics": {"loss": 0.25}, "step": 7})

        csv_files = list(run_manager.run_dir.glob("**/*.csv"))
        assert len(csv_files) > 0
        content = csv_files[0].read_text()
        assert "loss" in content
        assert "0.25" in content

    # fabric.teardown() doesn't exist in this version


def test_run_manager_context_manager_captures_stdout():
    """stdout written inside the context lands in stdout.txt."""
    fabric = create_fabric(accelerator="cpu", precision="32-true")
    fabric.launch()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_manager = RunManager(
            fabric=fabric,
            app_name="test_app",
            config={"seed": 42},
            output_base=Path(tmpdir),
        )

        with run_manager:
            print("hello from the run")

        stdout_file = run_manager.run_dir / "artifacts" / "stdout.txt"
        assert stdout_file.exists()
        assert "hello from the run" in stdout_file.read_text()

    # fabric.teardown() doesn't exist in this version
