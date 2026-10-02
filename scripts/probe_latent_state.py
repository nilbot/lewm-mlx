r"""Linear State Probing of Latent Geometry in Le World Model (MLX).

Evaluates whether the self-supervised JEPA visual representation space linearly decodes
physical environment states (2D end-effector coordinates [x, y] in Push-T) without
explicit coordinate supervision.

Mathematical Formulation:
    Given frozen encoder representations:
        $$\mathbf{z} = \operatorname{Encoder}(\mathbf{x}) \in \mathbb{R}^{D}$$
    A linear probe is fit via Ridge regression:
        $$\mathbf{W}^* = \arg\min_{\mathbf{W}} \|\mathbf{Z}\mathbf{W} - \mathbf{S}\|_F^2 + \alpha \|\mathbf{W}\|_F^2$$
    Evaluation measures out-of-sample coefficient of determination:
        $$R^2 = 1 - \frac{\sum_{i=1}^N \|\mathbf{s}_i - \hat{\mathbf{s}}_i\|_2^2}{\sum_{i=1}^N \|\mathbf{s}_i - \bar{\mathbf{s}}\|_2^2}$$
"""

from pathlib import Path
from typing import Dict, List, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from lewm_mlx.dataset import PushTMiniDataset
from lewm_mlx.jepa import JEPA
from lewm_mlx.module import ARPredictor, Embedder, MLP
from lewm_mlx.preprocessing import imagenet_normalize
from lewm_mlx.vit import ViTModel


def build_model(weights_path: str, img_size: int = 96, embed_dim: int = 64) -> JEPA:
    """Builds and loads a JEPA model from weights archive."""
    encoder = ViTModel(
        image_size=img_size,
        patch_size=16,
        num_channels=3,
        hidden_size=embed_dim,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
    )
    predictor = ARPredictor(
        num_frames=3,
        depth=2,
        heads=2,
        mlp_dim=256,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
    )
    action_encoder = Embedder(
        input_dim=10,
        smoothed_dim=embed_dim,
        emb_dim=embed_dim,
    )
    projector = MLP(
        input_dim=embed_dim,
        hidden_dim=256,
        output_dim=embed_dim,
        norm_fn=nn.BatchNorm,
    )
    pred_proj = MLP(
        input_dim=embed_dim,
        hidden_dim=256,
        output_dim=embed_dim,
        norm_fn=nn.BatchNorm,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.load_weights(weights_path)
    model.eval()
    return model


def load_aligned_states(num_episodes: int = 100, frameskip: int = 5) -> List[np.ndarray]:
    """Loads ground-truth continuous (x, y) coordinates aligned with subsampled macro-frames."""
    pf = hf_hub_download(
        repo_id="lerobot/pusht",
        filename="data/chunk-000/file-000.parquet",
        repo_type="dataset",
        local_files_only=True,
    )
    table = pq.read_table(pf)
    ep_indices = table["episode_index"].to_numpy()
    states_raw = np.array(table["observation.state"].to_pylist(), dtype=np.float32)

    unique_eps = np.unique(ep_indices)[:num_episodes]
    aligned = []
    for ep in unique_eps:
        mask = ep_indices == ep
        ep_s = states_raw[mask]
        n_macro = len(ep_s) // frameskip
        sub_s = ep_s[: n_macro * frameskip : frameskip]
        aligned.append(sub_s)
    return aligned


def extract_latent_representations(
    model: JEPA, dataset: PushTMiniDataset, num_episodes: int = 100
) -> List[np.ndarray]:
    """Encodes all frames into frozen latent representations [T_ep, D]."""
    representations = []
    for ep_idx in range(num_episodes):
        ep_p = dataset.episodes_pixels[ep_idx]  # [T, 3, H, W] in uint8
        p_norm = (ep_p.astype(np.float32) / 255.0)[np.newaxis, ...]  # [1, T, 3, H, W]
        p_tensor = imagenet_normalize(mx.array(p_norm))
        info = model.encode({"pixels": p_tensor})
        emb = np.array(info["emb"][0])  # [T, D]
        representations.append(emb)
    return representations


def run_probing() -> Dict[str, Dict[str, float]]:
    """Runs state probing benchmark across all checkpoints."""
    print("Loading Push-T dataset and ground-truth states...")
    num_episodes = 100
    dataset = PushTMiniDataset(num_episodes=num_episodes, frameskip=5, img_size=96)
    states = load_aligned_states(num_episodes=num_episodes, frameskip=5)

    # 70/30 train/test split at the episode level
    n_train_eps = 70
    train_states = np.concatenate(states[:n_train_eps], axis=0)  # [N_train, 2]
    test_states = np.concatenate(states[n_train_eps:], axis=0)    # [N_test, 2]

    checkpoints = {
        "sig_0.01": Path("outputs/experiments/lewm_sig_0.01.npz"),
        "sig_0.04": Path("outputs/experiments/lewm_sig_0.04.npz"),
        "sig_0.09": Path("outputs/experiments/lewm_sig_0.09.npz"),
        "sig_0.25": Path("outputs/experiments/lewm_sig_0.25.npz"),
        "sig_0.50": Path("outputs/experiments/lewm_sig_0.50.npz"),
        "k1": Path("outputs/experiments/lewm_k1.npz"),
        "k2": Path("outputs/experiments/lewm_k2.npz"),
        "k3": Path("outputs/experiments/lewm_k3.npz"),
        "ar_k3": Path("outputs/experiments/lewm_ar_k3.npz"),
        "comp_k3": Path("outputs/experiments/lewm_comp_k3.npz"),
    }

    results: Dict[str, Dict[str, float]] = {}

    print(f"\nEvaluating Linear State Probes on {len(test_states)} held-out test frames...")
    print(f"{'Model':<12} | {'Train R2':<10} | {'Test R2':<10} | {'Test MAE (px)':<14} | {'Test RMSE (px)':<14}")
    print("-" * 70)

    for name, ckpt in checkpoints.items():
        if not ckpt.exists():
            continue
        model = build_model(str(ckpt))
        embs = extract_latent_representations(model, dataset, num_episodes=num_episodes)

        train_z = np.concatenate(embs[:n_train_eps], axis=0)  # [N_train, D]
        test_z = np.concatenate(embs[n_train_eps:], axis=0)    # [N_test, D]

        # Fit Ridge regression
        probe = Ridge(alpha=1.0)
        probe.fit(train_z, train_states)

        train_pred = probe.predict(train_z)
        test_pred = probe.predict(test_z)

        train_r2 = float(r2_score(train_states, train_pred))
        test_r2 = float(r2_score(test_states, test_pred))
        test_mae = float(mean_absolute_error(test_states, test_pred))
        test_rmse = float(np.sqrt(np.mean(np.square(test_states - test_pred))))

        results[name] = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }

        print(
            f"{name:<12} | {train_r2:<10.4f} | {test_r2:<10.4f} | "
            f"{test_mae:<14.2f} | {test_rmse:<14.2f}"
        )

    return results


if __name__ == "__main__":
    run_probing()
