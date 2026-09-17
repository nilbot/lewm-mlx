"""Unit tests for PushTMiniDataset."""

from pathlib import Path
import mlx.core as mx

from lewm_mlx.dataset import PushTMiniDataset


def test_pusht_mini_dataset_offline_fallback(tmp_path: Path):
    """Verifies procedural fallback generation, batch shapes, types, and bounds."""
    dataset = PushTMiniDataset(
        cache_dir=str(tmp_path),
        num_episodes=2,
        frameskip=5,
        img_size=96,
        force_fallback=True,
    )
    assert dataset.num_episodes == 2
    assert len(dataset.episodes_pixels) == 2
    assert len(dataset.episodes_actions) == 2

    # Sample batch: batch_size=4, history_size=3, num_preds=2 -> T = 5
    batch = dataset.sample_batch(batch_size=4, history_size=3, num_preds=2)

    # Output dictionary keys
    assert "pixels" in batch
    assert "action" in batch

    # Check MLX tensor types
    assert isinstance(batch["pixels"], mx.array)
    assert isinstance(batch["action"], mx.array)

    # Check tensor shapes: [B, T, 3, H, W] and [B, T, frameskip * 2]
    assert batch["pixels"].shape == (4, 5, 3, 96, 96)
    assert batch["action"].shape == (4, 5, 10)

    # Value bounds: pixels in [0.0, 1.0]
    assert mx.min(batch["pixels"]).item() >= 0.0
    assert mx.max(batch["pixels"]).item() <= 1.0

    # No NaNs or Infs
    assert not mx.any(mx.isnan(batch["pixels"])).item()
    assert not mx.any(mx.isnan(batch["action"])).item()
    assert not mx.any(mx.isinf(batch["pixels"])).item()
    assert not mx.any(mx.isinf(batch["action"])).item()


def test_pusht_mini_dataset_padding(tmp_path: Path):
    """Verifies edge padding when requested sequence length exceeds episode length."""
    dataset = PushTMiniDataset(
        cache_dir=str(tmp_path),
        num_episodes=2,
        frameskip=5,
        img_size=96,
        force_fallback=True,
    )
    # Default fallback has T=40. Request T = 45 > 40
    batch = dataset.sample_batch(batch_size=2, history_size=25, num_preds=20)
    assert batch["pixels"].shape == (2, 45, 3, 96, 96)
    assert batch["action"].shape == (2, 45, 10)
    assert not mx.any(mx.isnan(batch["pixels"])).item()


def test_pusht_mini_dataset_download_and_caching(tmp_path: Path):
    """Tests downloading 2 episodes from lerobot/pusht and cache reloading."""
    dataset = PushTMiniDataset(
        cache_dir=str(tmp_path),
        num_episodes=2,
        frameskip=5,
        img_size=96,
    )
    assert dataset.num_episodes == 2
    assert len(dataset.episodes_pixels) == 2
    assert len(dataset.episodes_actions) == 2
    assert dataset.cache_path.exists()

    # Verify reloading from the cached .npz archive without download
    cached_dataset = PushTMiniDataset(
        cache_dir=str(tmp_path),
        num_episodes=2,
        frameskip=5,
        img_size=96,
    )
    assert cached_dataset.num_episodes == 2
    assert len(cached_dataset.episodes_pixels) == 2
    assert len(cached_dataset.episodes_actions) == 2

    # Verify sample_batch from cached data
    batch = cached_dataset.sample_batch(batch_size=2, history_size=2, num_preds=3)
    assert batch["pixels"].shape == (2, 5, 3, 96, 96)
    assert batch["action"].shape == (2, 5, 10)
    assert mx.min(batch["pixels"]).item() >= 0.0
    assert mx.max(batch["pixels"]).item() <= 1.0
    assert mx.min(batch["action"]).item() >= -1.0
    assert mx.max(batch["action"]).item() <= 1.0
    assert not mx.any(mx.isnan(batch["pixels"])).item()
    assert not mx.any(mx.isnan(batch["action"])).item()
