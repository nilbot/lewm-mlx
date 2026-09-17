"""Integration tests for unified Push-T demo CLI."""

import subprocess
import sys


def test_demo_cli_execution(tmp_path):
    """Verifies demo CLI executes both rollout and planning modes and outputs plot."""
    plot_path = tmp_path / "test_pusht_demo.png"
    cmd = [
        sys.executable,
        "demo.py",
        "--mode",
        "both",
        "--num-episodes",
        "2",
        "--horizon",
        "3",
        "--history-size",
        "2",
        "--img-size",
        "96",
        "--save-plot",
        str(plot_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"demo.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )
    assert "Multi-Step Autoregressive Rollout" in res.stdout
    assert "Goal-Directed Planning" in res.stdout
    assert plot_path.exists()


def test_demo_cli_rollout_mode(tmp_path):
    """Verifies demo CLI executes rollout mode only."""
    plot_path = tmp_path / "test_rollout.png"
    cmd = [
        sys.executable,
        "demo.py",
        "--mode",
        "rollout",
        "--num-episodes",
        "2",
        "--horizon",
        "2",
        "--history-size",
        "2",
        "--img-size",
        "96",
        "--save-plot",
        str(plot_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"demo.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )
    assert "Multi-Step Autoregressive Rollout" in res.stdout
    assert "Goal-Directed Planning" not in res.stdout
    assert plot_path.exists()


def test_demo_cli_plan_mode(tmp_path):
    """Verifies demo CLI executes planning mode only."""
    plot_path = tmp_path / "test_plan.png"
    cmd = [
        sys.executable,
        "demo.py",
        "--mode",
        "plan",
        "--num-episodes",
        "2",
        "--horizon",
        "2",
        "--history-size",
        "2",
        "--img-size",
        "96",
        "--save-plot",
        str(plot_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"demo.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )
    assert "Goal-Directed Planning" in res.stdout
    assert "Multi-Step Autoregressive Rollout" not in res.stdout
    assert plot_path.exists()


def test_demo_cli_with_weights(tmp_path):
    """Verifies demo CLI loads custom weights without error."""
    from demo import build_model

    weights_path = tmp_path / "custom_weights.npz"
    model = build_model(
        img_size=96,
        embed_dim=64,
        history_size=2,
        frameskip=5,
        weights_path=None,
    )
    model.save_weights(str(weights_path))

    plot_path = tmp_path / "test_weights_demo.png"
    cmd = [
        sys.executable,
        "demo.py",
        "--weights",
        str(weights_path),
        "--mode",
        "both",
        "--num-episodes",
        "2",
        "--horizon",
        "2",
        "--history-size",
        "2",
        "--img-size",
        "96",
        "--save-plot",
        str(plot_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"demo.py failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )
    assert f"Loading model weights from '{weights_path}'" in res.stdout
    assert plot_path.exists()


def test_demo_programmatic_api():
    """Verifies direct in-process programmatic execution of demo components."""
    import mlx.core as mx
    from demo import build_model, run_planning, run_rollout

    img_size = 96
    embed_dim = 32
    history_size = 2
    horizon = 2
    frameskip = 5
    action_dim = frameskip * 2
    planning_horizon = history_size + horizon

    model = build_model(
        img_size=img_size,
        embed_dim=embed_dim,
        history_size=history_size,
        frameskip=frameskip,
        weights_path=None,
    )

    # Batch with num_preds = horizon + 1 (total length planning_horizon + 1)
    seq_len = planning_horizon + 1
    batch = {
        "pixels": mx.random.normal((1, seq_len, 3, img_size, img_size)),
        "action": mx.random.normal((1, seq_len, action_dim)),
    }

    step_indices, mse_errors = run_rollout(
        model=model,
        batch=batch,
        history_size=history_size,
        horizon=horizon,
    )
    assert len(step_indices) == horizon
    assert len(mse_errors) == horizon
    assert all(isinstance(err, float) for err in mse_errors)

    shooting_cost, cem_cost, cem_cost_history = run_planning(
        model=model,
        batch=batch,
        history_size=history_size,
        horizon=horizon,
        action_dim=action_dim,
    )
    assert isinstance(shooting_cost, float)
    assert isinstance(cem_cost, float)
    assert len(cem_cost_history) == 5
    assert cem_cost <= shooting_cost + 1e-5


def test_demo_cli_validation():
    """Verifies demo CLI rejects non-positive parameters."""
    cmd = [
        sys.executable,
        "demo.py",
        "--history-size",
        "0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "ValueError: history_size must be positive" in res.stderr
