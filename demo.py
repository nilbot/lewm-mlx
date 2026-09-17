"""Unified Demonstration CLI for Le World Model in MLX.

Executes multi-step autoregressive latent rollouts and cross-entropy method (CEM)
goal-directed planning on demonstration trajectories from the Push-T manipulation
environment, producing tabular error metrics and visual multi-panel diagnostic plots.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Set non-interactive headless backend prior to pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lewm_mlx.dataset import PushTMiniDataset
from lewm_mlx.jepa import JEPA
from lewm_mlx.module import ARPredictor, Embedder, MLP
from lewm_mlx.planner import CEMPlanner, ShootingPlanner
from lewm_mlx.vit import ViTModel


def build_model(
    img_size: int,
    embed_dim: int,
    history_size: int,
    frameskip: int,
    weights_path: Optional[str] = None,
    explicit_weights: bool = False,
) -> JEPA:
    r"""Builds and initializes the Joint-Embedding Predictive Architecture (JEPA).

    Instantiates a Vision Transformer (ViTModel) patch encoder, continuous action
    sequence embedder, multi-head autoregressive transformer predictor, and projection
    MLPs. Loads pre-trained weights if available at ``weights_path``.

    Architecture:
        - ViT Visual Encoder: :math:`E_{\theta}(\mathbf{x}) \in \mathbb{R}^D`
        - Action Embedder: :math:`\phi_{\psi}(\mathbf{a}) \in \mathbb{R}^D`
        - AR Latent Predictor: :math:`P_{\phi}(\mathbf{z}_{t-H:t}, \mathbf{a}_{t-H:t}) \to \hat{\mathbf{z}}_{t+1}`

    Args:
        img_size: Square image spatial resolution (:math:`H_{\text{img}} = W_{\text{img}}`).
        embed_dim: Latent embedding dimension (:math:`D`).
        history_size: Temporal length of observation context window (:math:`H`).
        frameskip: Action chunking and downsampling factor (:math:`F`).
        weights_path: Optional path to serialized model weights archive (.npz).
        explicit_weights: Whether the weights path was explicitly supplied by user.

    Returns:
        Configured JEPA model set to evaluation mode.
    """
    action_dim = frameskip * 2

    # ViT visual encoder for patch-based representation
    encoder = ViTModel(
        image_size=img_size,
        patch_size=16,
        num_channels=3,
        hidden_size=embed_dim,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
    )

    # Autoregressive transformer predictor in latent space
    predictor = ARPredictor(
        num_frames=history_size,
        depth=2,
        heads=2,
        mlp_dim=256,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
    )

    # Action encoder projecting continuous control chunks to latent embedding space
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=embed_dim,
        emb_dim=embed_dim,
    )

    # Latent projection heads
    projector = MLP(
        input_dim=embed_dim,
        hidden_dim=256,
        output_dim=embed_dim,
        norm_fn=nn.LayerNorm,
    )
    pred_proj = MLP(
        input_dim=embed_dim,
        hidden_dim=256,
        output_dim=embed_dim,
        norm_fn=nn.LayerNorm,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.eval()

    is_user_specified = explicit_weights or (
        weights_path is not None and weights_path != "lewm_weights.npz"
    )

    if weights_path and os.path.exists(weights_path):
        print(f"Loading model weights from '{weights_path}'...")
        try:
            model.load_weights(weights_path)
        except Exception as exc:
            # Shape mismatches commonly occur when pre-trained position embeddings
            # (e.g. 197 tokens from 224x224 resolution: (224/16)^2 + 1 = 197) do not
            # match the current spatial token count (e.g. 37 tokens from 96x96: (96/16)^2 + 1 = 37).
            if is_user_specified:
                raise RuntimeError(
                    f"Failed to load user-specified weights from {weights_path}: {exc}"
                ) from exc
            print(
                f"Warning: Failed to load default weights from '{weights_path}': {exc}. "
                "Continuing with initialized model for zero-setup demo."
            )
    elif is_user_specified and weights_path:
        raise FileNotFoundError(
            f"User-specified weights file '{weights_path}' not found."
        )
    else:
        print(
            f"Notice: Weights file '{weights_path}' not found. "
            "Continuing with initialized model for zero-setup demo."
        )

    return model


def run_rollout(
    model: JEPA,
    batch: Dict[str, mx.array],
    history_size: int,
    horizon: int,
) -> Tuple[List[int], List[float]]:
    r"""Executes multi-step autoregressive latent rollout and evaluates MSE error.

    Rolls out latent predictions autoregressively conditioned on past context frames
    :math:`\mathbf{s}_{0:H}` and ground-truth action sequence :math:`\mathbf{A}_{0:H+K}`:

    .. math::
        \hat{\mathbf{z}}_{t} = P_\phi(\hat{\mathbf{z}}_{t-H:t-1}, \mathbf{a}_{t-H:t-1})
        \quad \text{for } t \in [H, H+K)

    and computes the step-by-step Mean Squared Error relative to encoded ground-truth frames:

    .. math::
        \text{MSE}_t = \frac{1}{D} \left\| \hat{\mathbf{z}}_t - \mathbf{z}_t \right\|_2^2

    Args:
        model: Evaluated JEPA model instance.
        batch: Demonstration trajectory dictionary with "pixels" and "action" keys.
        history_size: Context history length (:math:`H`).
        horizon: Prediction forward horizon (:math:`K`).

    Returns:
        Tuple of (step_indices, mse_errors):
            step_indices: Sequence time steps :math:`t \in [H, H+K)`.
            mse_errors: Step-by-step mean squared error scalars.
    """
    pixels = batch["pixels"]  # [1, T, 3, H_img, W_img]
    actions = batch["action"]  # [1, T, action_dim]
    actions = mx.where(mx.isnan(actions), mx.array(0.0), actions)

    # Initial context history: [1, 1, H, 3, H_img, W_img]
    context_pixels = pixels[:, :history_size]
    rollout_info = {"pixels": mx.expand_dims(context_pixels, 1)}
    # Action trajectory up to planning horizon T = history_size + horizon
    action_seq_len = history_size + horizon
    rollout_actions = mx.expand_dims(actions[:, :action_seq_len], 1)

    # Execute latent autoregressive rollout
    rollout_out = model.rollout(
        rollout_info, rollout_actions, history_size=history_size
    )
    pred_emb = rollout_out["predicted_emb"]  # [1, 1, T_pred, D]

    # Encode ground-truth frames across trajectory: [1, T, D]
    gt_out = model.encode({"pixels": pixels[:, :action_seq_len]})
    gt_emb = gt_out["emb"]  # [1, T, D]

    print()
    print("=" * 60)
    print("         Multi-Step Autoregressive Rollout")
    print("=" * 60)
    print(f"{'Step':<8}{'Target norm':<15}{'Pred norm':<15}{'MSE':<15}")
    print("-" * 60)

    step_indices: List[int] = []
    mse_errors: List[float] = []

    for k in range(horizon):
        t = history_size + k
        tgt_vec = gt_emb[0, t]  # [D]
        pred_vec = pred_emb[0, 0, t]  # [D]

        tgt_norm = mx.linalg.norm(tgt_vec).item()
        pred_norm = mx.linalg.norm(pred_vec).item()
        mse = mx.mean(mx.square(pred_vec - tgt_vec)).item()

        step_indices.append(t)
        mse_errors.append(mse)

        print(f"{t:<8}{tgt_norm:<15.4f}{pred_norm:<15.4f}{mse:<15.6f}")

    print("=" * 60)
    return step_indices, mse_errors


def run_planning(
    model: JEPA,
    batch: Dict[str, mx.array],
    history_size: int,
    horizon: int,
    action_dim: int,
) -> Tuple[float, float, List[float]]:
    r"""Executes goal-directed planning using Random Shooting baseline and CEM.

    Optimizes open-loop action sequences to minimize distance between predicted final
    latent state :math:`\hat{\mathbf{z}}_{T}` and encoded distant goal :math:`\mathbf{z}_{\text{goal}}`:

    .. math::
        \min_{\mathbf{A}} \mathcal{J}(\mathbf{A}) = \left\| \hat{\mathbf{z}}_T(\mathbf{A}) - \mathbf{z}_{\text{goal}} \right\|_2^2

    Args:
        model: Evaluated JEPA model instance.
        batch: Trajectory dictionary with "pixels" key containing at least $T + 1$ frames.
        history_size: Context history length (:math:`H`).
        horizon: Planning prediction horizon (:math:`K`).
        action_dim: Dimensionality of action vectors (:math:`D_{\text{act}} = 2 \times F`).

    Returns:
        Tuple of (shooting_cost, cem_cost, cem_cost_history):
            shooting_cost: Objective cost achieved by baseline random shooting.
            cem_cost: Objective cost achieved by cross-entropy method optimizer.
            cem_cost_history: Cost trace across CEM iterations.
    """
    pixels = batch["pixels"]  # [1, T + 1, 3, H_img, W_img]
    planning_horizon = history_size + horizon  # T

    # Context observation frames: [1, 1, H, 3, H_img, W_img]
    init_pixels = mx.expand_dims(pixels[:, :history_size], 1)
    # Distant target goal observation frame at t = planning_horizon: [1, 1, 1, 3, H_img, W_img]
    goal_pixels = mx.expand_dims(pixels[:, planning_horizon : planning_horizon + 1], 1)

    plan_info = {
        "pixels": init_pixels,
        "goal": goal_pixels,
    }

    print()
    print("=" * 60)
    print("               Goal-Directed Planning")
    print("=" * 60)

    # 1. Baseline: Random Shooting Planner
    shooting_planner = ShootingPlanner(
        model=model,
        planning_horizon=planning_horizon,
        action_dim=action_dim,
        num_samples=64,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    _, shooting_cost, _ = shooting_planner.plan(dict(plan_info))

    # 2. Cross-Entropy Method (CEM) Planner
    cem_planner = CEMPlanner(
        model=model,
        planning_horizon=planning_horizon,
        action_dim=action_dim,
        num_samples=64,
        num_elites=8,
        iterations=5,
        alpha=0.1,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    _, cem_cost, cem_cost_history = cem_planner.plan(dict(plan_info))

    cost_reduction = (
        ((shooting_cost - cem_cost) / shooting_cost) * 100.0
        if shooting_cost > 0
        else 0.0
    )

    print(f"Random Shooting Cost (Baseline): {shooting_cost:.6f}")
    print(f"Cross-Entropy Method Cost (CEM): {cem_cost:.6f}")
    print(f"CEM Optimization Cost Reduction: {cost_reduction:.2f}%")
    print("=" * 60)

    return shooting_cost, cem_cost, cem_cost_history


def save_diagnostic_plot(
    save_path: str,
    batch: Dict[str, mx.array],
    history_size: int,
    horizon: int,
    goal_frame: Optional[Any] = None,
    step_indices: Optional[List[int]] = None,
    mse_errors: Optional[List[float]] = None,
    shooting_cost: Optional[float] = None,
    cem_cost_history: Optional[List[float]] = None,
) -> None:
    """Generates and saves multi-panel diagnostic plot summarizing demo results.

    Creates a 2x3 subplot grid containing:
        - Top row: Observation context frames (:math:`t = 0, 1, 2`).
        - Bottom-left: Target goal frame (:math:`s_{T}`).
        - Bottom-center: Latent rollout prediction MSE error trajectory curve.
        - Bottom-right: CEM planning cost convergence curve over iterations.

    Args:
        save_path: Filesystem path to output PNG image.
        batch: Trajectory dictionary with "pixels" MLX array.
        history_size: Context history length (:math:`H`).
        horizon: Prediction forward horizon (:math:`K`).
        goal_frame: Optional goal frame array to display at bottom-left.
        step_indices: Optional rollout sequence step indices.
        mse_errors: Optional step-by-step rollout MSE errors.
        shooting_cost: Optional baseline random shooting objective cost.
        cem_cost_history: Optional list of CEM costs across refinement iterations.
    """
    pixels = np.array(batch["pixels"])  # [1, T, 3, H_img, W_img]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Top row: Context frames (t = 0, 1, 2)
    for col in range(3):
        ax = axes[0, col]
        if col < history_size and col < pixels.shape[1]:
            # Transpose [3, H, W] -> [H, W, 3] and normalize to [0, 1]
            frame = np.clip(pixels[0, col].transpose(1, 2, 0), 0.0, 1.0)
            ax.imshow(frame)
            ax.set_title(f"Context Frame t={col}", fontsize=11, fontweight="bold")
        else:
            ax.text(
                0.5,
                0.5,
                "(No Frame)",
                ha="center",
                va="center",
                color="gray",
            )
            ax.set_title(f"Context Frame t={col}", fontsize=11)
        ax.axis("off")

    # Bottom-left: Goal frame at t = planning_horizon
    planning_horizon = history_size + horizon
    ax_goal = axes[1, 0]
    if goal_frame is not None:
        gf = np.clip(np.array(goal_frame).transpose(1, 2, 0), 0.0, 1.0)
        ax_goal.imshow(gf)
        ax_goal.set_title(
            f"Goal Frame (t={planning_horizon})", fontsize=11, fontweight="bold"
        )
    elif planning_horizon < pixels.shape[1]:
        gf = np.clip(pixels[0, planning_horizon].transpose(1, 2, 0), 0.0, 1.0)
        ax_goal.imshow(gf)
        ax_goal.set_title(
            f"Goal Frame (t={planning_horizon})", fontsize=11, fontweight="bold"
        )
    else:
        ax_goal.text(
            0.5, 0.5, "(Goal Out of Range)", ha="center", va="center", color="gray"
        )
        ax_goal.set_title("Goal Frame", fontsize=11)
    ax_goal.axis("off")

    # Bottom-center: Latent error trajectory curve
    ax_rollout = axes[1, 1]
    if step_indices and mse_errors:
        ax_rollout.plot(
            step_indices,
            mse_errors,
            marker="o",
            linewidth=2,
            color="crimson",
            label="Latent MSE",
        )
        ax_rollout.set_xlabel("Rollout Step (t)", fontsize=10)
        ax_rollout.set_ylabel(r"Latent MSE $\|\hat{z}_t - z_t\|^2$", fontsize=10)
        ax_rollout.set_title("Rollout Latent Error", fontsize=11, fontweight="bold")
        ax_rollout.grid(True, linestyle="--", alpha=0.6)
        ax_rollout.legend(loc="upper left")
    else:
        ax_rollout.text(
            0.5,
            0.5,
            "Rollout Mode Skipped",
            ha="center",
            va="center",
            color="gray",
        )
        ax_rollout.set_title("Rollout Latent Error", fontsize=11)
        ax_rollout.axis("off")

    # Bottom-right: CEM cost convergence curve
    ax_plan = axes[1, 2]
    if cem_cost_history:
        iterations = list(range(1, len(cem_cost_history) + 1))
        ax_plan.plot(
            iterations,
            cem_cost_history,
            marker="s",
            linewidth=2,
            color="royalblue",
            label="CEM Cost",
        )
        if shooting_cost is not None:
            ax_plan.axhline(
                shooting_cost,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label=f"Shooting ({shooting_cost:.3f})",
            )
        ax_plan.set_xlabel("Iteration", fontsize=10)
        ax_plan.set_ylabel("Objective Cost", fontsize=10)
        ax_plan.set_title("CEM Cost Convergence", fontsize=11, fontweight="bold")
        ax_plan.grid(True, linestyle="--", alpha=0.6)
        ax_plan.legend(loc="upper right")
    else:
        ax_plan.text(
            0.5,
            0.5,
            "Planning Mode Skipped",
            ha="center",
            va="center",
            color="gray",
        )
        ax_plan.set_title("CEM Cost Convergence", fontsize=11)
        ax_plan.axis("off")

    plt.tight_layout()
    plot_dir = os.path.dirname(os.path.abspath(save_path))
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization plot to '{save_path}'")


def main() -> None:
    """Parses arguments and runs unified demonstration pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Demo CLI for Le World Model in MLX"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="lewm_weights.npz",
        help="Path to pre-trained weights archive (.npz)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rollout", "plan", "both"],
        default="both",
        help="Evaluation mode: rollout, plan, or both",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of demonstration episodes to load",
    )
    parser.add_argument(
        "--frameskip",
        type=int,
        default=5,
        help="Action chunking and frameskip factor",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        help="Square input image resolution",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Latent embedding dimension",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=3,
        help="Context observation history length",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction / planning future horizon",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="pusht_demo.png",
        help="Destination path for visual multi-panel diagnostic plot",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local directory for dataset cache archive",
    )

    args = parser.parse_args()

    # Input validation
    if args.history_size <= 0:
        raise ValueError(f"history_size must be positive, got {args.history_size}")
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    if args.num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {args.num_episodes}")
    if args.frameskip <= 0:
        raise ValueError(f"frameskip must be positive, got {args.frameskip}")

    # Build model components
    explicit_weights = "--weights" in sys.argv
    model = build_model(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        frameskip=args.frameskip,
        weights_path=args.weights,
        explicit_weights=explicit_weights,
    )

    # Load demonstration dataset
    dataset = PushTMiniDataset(
        cache_dir=args.cache_dir,
        num_episodes=args.num_episodes,
        frameskip=args.frameskip,
        img_size=args.img_size,
    )

    # Sample trajectory sequence with num_preds = horizon + 1 for exact temporal alignment
    batch = dataset.sample_batch(
        batch_size=1,
        history_size=args.history_size,
        num_preds=args.horizon + 1,
    )

    planning_horizon = args.history_size + args.horizon
    goal_frame = batch["pixels"][0, planning_horizon]

    step_indices: Optional[List[int]] = None
    mse_errors: Optional[List[float]] = None
    shooting_cost: Optional[float] = None
    cem_cost_history: Optional[List[float]] = None

    if args.mode in ("rollout", "both"):
        step_indices, mse_errors = run_rollout(
            model=model,
            batch=batch,
            history_size=args.history_size,
            horizon=args.horizon,
        )

    if args.mode in ("plan", "both"):
        action_dim = args.frameskip * 2
        shooting_cost, _, cem_cost_history = run_planning(
            model=model,
            batch=batch,
            history_size=args.history_size,
            horizon=args.horizon,
            action_dim=action_dim,
        )

    if args.save_plot:
        save_diagnostic_plot(
            save_path=args.save_plot,
            batch=batch,
            history_size=args.history_size,
            horizon=args.horizon,
            goal_frame=goal_frame,
            step_indices=step_indices,
            mse_errors=mse_errors,
            shooting_cost=shooting_cost,
            cem_cost_history=cem_cost_history,
        )


if __name__ == "__main__":
    main()
