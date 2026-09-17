import subprocess
import sys


def test_train_cli_pusht_mini(tmp_path):
    """Verifies that lewm_mlx.train CLI executes properly with pusht_mini dataset."""
    save_path = tmp_path / "test_lewm_weights.npz"
    cmd = [
        sys.executable,
        "-m",
        "lewm_mlx.train",
        "--dataset",
        "pusht_mini",
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
