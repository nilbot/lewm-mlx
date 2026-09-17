"""PushT Mini Dataset Loader and Generator in MLX.

Downloads, extracts, caches, and samples demonstration episodes from
the Push-T manipulation environment for Le World Model training and evaluation.
Provides offline fallback to procedural synthetic trajectory generation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import mlx.core as mx
import numpy as np


def _as_uint8_frames(frames: np.ndarray) -> np.ndarray:
    """Converts cached ``[0, 1]`` float frames to uint8 storage.

    Args:
        frames: Frame array of shape :math:`[T, 3, H, W]` in either uint8 or
            float32 :math:`[0, 1]` representation.

    Returns:
        uint8 frame array; float inputs are scaled by 255 and rounded, which is
        exact for frames originally derived from 8-bit images.
    """
    if frames.dtype == np.uint8:
        return frames
    return np.clip(frames * 255.0, 0.0, 255.0).round().astype(np.uint8)


class PushTMiniDataset:
    """Dataset loader for Push-T demonstration trajectories.

    Extracts demonstration episodes from lerobot/pusht on Hugging Face Hub,
    resizes visual observations to square resolution, normalizes end-effector
    actions to :math:`[-1.0, 1.0]`, and groups consecutive actions by frameskip.
    Provides offline procedural synthetic fallback when network is unavailable.

    Attributes:
        cache_dir: Directory where the parsed .npz archive is stored.
        num_episodes: Number of demonstration episodes to extract.
        frameskip: Action frameskip grouping factor :math:`F`.
        img_size: Target square image dimension :math:`(H, W)`.
        action_dim: Dimension of concatenated macro-actions :math:`2 \\times F`.
        cache_path: Path to the compressed cache archive.
        episodes_pixels: List of processed frame arrays, each of shape
            :math:`[T_{\\text{macro}}, 3, H, W]`.
        episodes_actions: List of chunked action arrays, each of shape
            :math:`[T_{\\text{macro}}, 2 \\times F]`.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        num_episodes: int = 10,
        frameskip: int = 5,
        img_size: int = 96,
        force_download: bool = False,
        force_fallback: bool = False,
    ) -> None:
        """Initializes PushTMiniDataset with local caching or offline fallback.

        Args:
            cache_dir: Local filesystem directory for storing cached .npz files.
                Defaults to ~/.cache/lewm.
            num_episodes: Number of demonstration episodes to extract and cache.
            frameskip: Frame subsampling rate and action chunking factor.
            img_size: Target square image resolution (height and width).
            force_download: If True, re-downloads and re-processes from HF Hub.
            force_fallback: If True, skips HF download and uses synthetic data.

        Raises:
            ValueError: If num_episodes, frameskip, or img_size are not positive.
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        if frameskip <= 0:
            raise ValueError(f"frameskip must be positive, got {frameskip}")
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/lewm")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.num_episodes = num_episodes
        self.frameskip = frameskip
        self.img_size = img_size
        # 2D continuous end-effector coordinates (x, y) concatenated over frameskip
        # D_act = 2 * frameskip
        self.action_dim = frameskip * 2

        self.cache_path = (
            self.cache_dir / f"pusht_mini_{num_episodes}ep_f{frameskip}_{img_size}px.npz"
        )

        self.episodes_pixels: List[np.ndarray] = []
        self.episodes_actions: List[np.ndarray] = []

        if force_fallback:
            self._load_fallback()
            return

        loaded = False
        if self.cache_path.exists() and not force_download:
            try:
                self._load_from_cache()
                loaded = True
            except Exception as e:
                print(
                    f"Warning: Failed to load cache from {self.cache_path} ({e}). "
                    "Removing corrupted cache and re-processing."
                )
                try:
                    self.cache_path.unlink(missing_ok=True)
                except OSError:
                    pass

        if not loaded:
            try:
                self._download_and_process()
            except Exception as e:
                print(
                    f"Warning: Failed to download lerobot/pusht ({e}). "
                    "Falling back to procedural synthetic data."
                )
                self._load_fallback()

    def _download_and_process(self) -> None:
        """Downloads Chunk-0 from Hugging Face and extracts demonstration episodes.

        Downloads parquet telemetry and MP4 video from `lerobot/pusht`, normalizes
        actions via:
            $$\\mathbf{a}_{\\text{norm}} = \\frac{\\mathbf{a}_{\\text{raw}}}{256.0} - 1.0 \\in [-1.0, 1.0]$$
        and groups frames and actions by frameskip :math:`F`:
            $$\\mathbf{A}_t = [\\mathbf{a}_{t \\cdot F}, \\dots, \\mathbf{a}_{(t+1) \\cdot F - 1}] \\in \\mathbb{R}^{2F}$$
        """
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        parquet_file = hf_hub_download(
            repo_id="lerobot/pusht",
            filename="data/chunk-000/file-000.parquet",
            repo_type="dataset",
        )
        video_file = hf_hub_download(
            repo_id="lerobot/pusht",
            filename="videos/observation.image/chunk-000/file-000.mp4",
            repo_type="dataset",
        )

        table = pq.read_table(parquet_file)
        ep_indices = table["episode_index"].to_numpy()  # [Total_Rows]
        actions_raw = np.array(table["action"].to_pylist(), dtype=np.float32)  # [Total_Rows, 2]

        # Normalize actions to [-1.0, 1.0] from raw Push-T [0, 512] coordinates
        # a_norm = (a_raw / 256.0) - 1.0
        actions_norm = (actions_raw / 256.0) - 1.0  # [Total_Rows, 2]

        unique_eps = np.unique(ep_indices)[: self.num_episodes]
        # Find maximum frame index required to avoid reading unnecessary frames
        max_frame = int(np.where(np.isin(ep_indices, unique_eps))[0][-1]) + 1

        # Read video frames via OpenCV
        cap = cv2.VideoCapture(video_file)
        frames = []
        frame_idx = 0
        while frame_idx < max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert OpenCV BGR to RGB: [H_orig, W_orig, 3]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize if necessary: [H, W, 3]
            if frame_rgb.shape[0] != self.img_size or frame_rgb.shape[1] != self.img_size:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (self.img_size, self.img_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            # Layout [3, H, W] stored as uint8; sample_batch restores [0.0, 1.0]
            frame_chw = np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
            frames.append(frame_chw)
            frame_idx += 1
        cap.release()

        frames_arr = np.stack(frames, axis=0)  # [N_frames, 3, H, W]

        # Extract first num_episodes
        episodes_pixels: List[np.ndarray] = []
        episodes_actions: List[np.ndarray] = []

        for ep in unique_eps:
            mask = ep_indices[: len(frames_arr)] == ep
            ep_frames = frames_arr[mask]  # [T_ep, 3, H, W]
            ep_acts = actions_norm[: len(frames_arr)][mask]  # [T_ep, 2]

            # Downsample frames and chunk actions by frameskip F
            n_macro = len(ep_acts) // self.frameskip
            if n_macro < 2:
                continue

            # Subsampled frames: [T_macro, 3, H, W]
            sub_frames = ep_frames[: n_macro * self.frameskip : self.frameskip]
            # Chunked actions: [T_macro, 2 * F]
            act_chunked = ep_acts[: n_macro * self.frameskip].reshape(
                n_macro, self.frameskip * 2
            )

            episodes_pixels.append(sub_frames)
            episodes_actions.append(act_chunked)

        self.episodes_pixels = episodes_pixels
        self.episodes_actions = episodes_actions

        # Save to compressed cache archive
        np.savez_compressed(
            self.cache_path,
            num_episodes=len(episodes_pixels),
            **{f"pixels_{i}": ep_p for i, ep_p in enumerate(episodes_pixels)},
            **{f"actions_{i}": ep_a for i, ep_a in enumerate(episodes_actions)},
        )

    def _load_from_cache(self) -> None:
        """Loads cached demonstration trajectories from local .npz archive."""
        with np.load(self.cache_path) as data:
            n = int(data["num_episodes"])
            self.episodes_pixels = [
                _as_uint8_frames(data[f"pixels_{i}"]) for i in range(n)
            ]
            self.episodes_actions = [data[f"actions_{i}"] for i in range(n)]

    def _load_fallback(self) -> None:
        """Generates synthetic 2D particle trajectories when offline.

        Constructs randomized smooth continuous trajectories mimicking
        manipulation episodes with bounded pixels :math:`[0.1, 0.9]` and
        normalized actions :math:`[-1.0, 1.0]`.
        """
        self.episodes_pixels = []
        self.episodes_actions = []
        t_steps = 40
        for _ in range(self.num_episodes):
            # Synthetic visual frames: [T, 3, H, W] stored as uint8 within ~[0.1, 0.9]
            p = (
                np.random.uniform(
                    0.1, 0.9, size=(t_steps, 3, self.img_size, self.img_size)
                )
                * 255.0
            ).round().astype(np.uint8)
            # Synthetic actions: [T, D_act] in [-1.0, 1.0]
            a = np.random.uniform(-1.0, 1.0, size=(t_steps, self.action_dim)).astype(
                np.float32
            )
            self.episodes_pixels.append(p)
            self.episodes_actions.append(a)

    def sample_batch(
        self, batch_size: int, history_size: int, num_preds: int
    ) -> Dict[str, mx.array]:
        """Samples a randomized sub-trajectory batch across episodes.

        Extracts continuous sub-trajectories of temporal length
        :math:`T = \\text{history\\_size} + \\text{num\\_preds}` from randomly
        selected demonstration episodes. If an episode is shorter than :math:`T`,
        the trajectory is padded at the boundary using edge replication.

        Args:
            batch_size: Number of parallel sequences in batch (:math:`B`).
            history_size: Number of past context frames (:math:`H`).
            num_preds: Number of future prediction horizons (:math:`K`).

        Returns:
            Dict containing:
                "pixels": MLX array of shape
                    :math:`[B, H + K, 3, \\text{img\\_size}, \\text{img\\_size}]`
                    in range :math:`[0.0, 1.0]`.
                "action": MLX array of shape
                    :math:`[B, H + K, \\text{action\\_dim}]`
                    in range :math:`[-1.0, 1.0]`.
        """
        seq_len = history_size + num_preds  # T = H + K
        batch_pixels: List[np.ndarray] = []
        batch_actions: List[np.ndarray] = []

        num_available = len(self.episodes_pixels)
        for _ in range(batch_size):
            ep_idx = np.random.randint(num_available)
            ep_p = self.episodes_pixels[ep_idx]  # [T_ep, 3, H, W]
            ep_a = self.episodes_actions[ep_idx]  # [T_ep, D_act]

            max_start = len(ep_p) - seq_len
            if max_start < 0:
                pad_len = -max_start
                # Replicate edge frames along temporal dimension
                p_slice = np.pad(
                    ep_p, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode="edge"
                )  # [T, 3, H, W]
                a_slice = np.pad(ep_a, ((0, pad_len), (0, 0)), mode="edge")  # [T, D_act]
            else:
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                p_slice = ep_p[start_idx : start_idx + seq_len]  # [T, 3, H, W]
                a_slice = ep_a[start_idx : start_idx + seq_len]  # [T, D_act]

            batch_pixels.append(p_slice)
            batch_actions.append(a_slice)

        # Output shape annotations:
        # pixels: [B, T, 3, H, W]
        # action: [B, T, D_act]
        pixels_batch = np.stack(batch_pixels, axis=0)  # [B, T, 3, H, W]
        actions_batch = np.stack(batch_actions, axis=0)  # [B, T, D_act]
        if pixels_batch.dtype == np.uint8:
            pixels_batch = pixels_batch.astype(np.float32) / 255.0
        else:
            pixels_batch = pixels_batch.astype(np.float32)

        return {
            "pixels": mx.array(pixels_batch),
            "action": mx.array(actions_batch),
        }
