"""Integration tests for the Le World Model training CLI in MLX.

Tests end-to-end execution of lewm_mlx.train across multiple dataset backends
including PushTMiniDataset and synthetic trajectory generation.
"""

import subprocess
import sys
import pytest


@pytest.mark.parametrize("dataset", ["pusht_mini", "synthetic"])
def test_train_cli(tmp_path, dataset):
    """Verifies that lewm_mlx.train CLI executes properly with the specified dataset.

    Args:
        tmp_path: Pytest temporary directory fixture.
        dataset: Name of dataset backend to evaluate ("pusht_mini" or "synthetic").
    """
    save_path = tmp_path / f"test_lewm_weights_{dataset}.npz"
    cache_dir = tmp_path / "cache"
    cmd = [
        sys.executable,
        "-m",
        "lewm_mlx.train",
        "--dataset",
        dataset,
        "--cache-dir",
        str(cache_dir),
        "--num-episodes",
        "2",
        "--epochs",
        "1",
        "--steps-per-epoch",
        "2",
        "--img-size",
        "96",
        "--save-path",
        str(save_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Training failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )
    assert f"Saving weights to {save_path}" in res.stdout
    assert save_path.exists()


def test_train_telemetry_options(tmp_path):
    """Verifies that diagnostic instrumentation and metrics logging function end-to-end.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    import json

    save_path = tmp_path / "telemetry_model.npz"
    metrics_path = tmp_path / "metrics.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "lewm_mlx.train",
        "--dataset",
        "synthetic",
        "--epochs",
        "2",
        "--steps-per-epoch",
        "2",
        "--img-size",
        "96",
        "--grad-breakdown",
        "--eval-fixed",
        "--log-interval",
        "1",
        "--metrics-path",
        str(metrics_path),
        "--save-path",
        str(save_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Training with telemetry failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )
    assert "CosSim:" in res.stdout
    assert "GradNorm:" in res.stdout
    assert "(enc:" in res.stdout
    assert "ValLoss:" in res.stdout
    assert metrics_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    for record in lines:
        assert "loss" in record
        assert "pred_loss" in record
        assert "cos_sim" in record
        assert "grad_norm" in record
        assert "enc_grad_norm" in record
        assert "val_pred_loss" in record
