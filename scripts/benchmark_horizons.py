r"""Benchmark Multi-Horizon Autoregressive Dynamics and Planning in MLX.

Compares model dynamics fidelity across:
1. Single-step model (k1)
2. Direct jump models (k2, k3)
3. Autoregressive BPTT unrolled model (ar_k3)
4. Teacher-forced composite model (comp_k3)

Metrics:
- Step-by-step rollout MSE and Cosine Similarity for horizons $k \in \{1, 2, 3, 4, 5, 6, 7, 8\}$
- Cross-Entropy Method (CEM) 5-step goal-directed planning cost reduction and terminal error
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import mlx.core as mx
import numpy as np

from lewm_mlx.dataset import PushTMiniDataset
from lewm_mlx.jepa import JEPA
from lewm_mlx.planner import CEMPlanner, ShootingPlanner
from lewm_mlx.preprocessing import imagenet_normalize
from scripts.probe_latent_state import build_model


def evaluate_rollout_horizons(
    model: JEPA,
    dataset: PushTMiniDataset,
    eval_episodes: List[int],
    history_size: int = 3,
    max_horizon: int = 8,
) -> Dict[int, Dict[str, float]]:
    """Evaluates autoregressive rollout error step-by-step up to max_horizon.

    Args:
        model: JEPA instance.
        dataset: PushTMiniDataset instance.
        eval_episodes: Indices of episodes used for evaluation.
        history_size: Number of context frames (H).
        max_horizon: Maximum future steps (K).

    Returns:
        Dict mapping step k (1..max_horizon) to {"mse": float, "cos_sim": float}.
    """
    seq_len = history_size + max_horizon
    step_errors: Dict[int, List[float]] = {k: [] for k in range(1, max_horizon + 1)}
    step_cosines: Dict[int, List[float]] = {k: [] for k in range(1, max_horizon + 1)}

    for ep_idx in eval_episodes:
        ep_pixels = dataset.episodes_pixels[ep_idx]
        ep_actions = dataset.episodes_actions[ep_idx]
        if len(ep_pixels) < seq_len:
            continue

        p_slice = (ep_pixels[:seq_len].astype(np.float32) / 255.0)[np.newaxis, ...]  # [1, T, 3, H, W]
        a_slice = ep_actions[:seq_len][np.newaxis, ...]  # [1, T, D_act]

        p_tensor = imagenet_normalize(mx.array(p_slice))
        a_tensor = mx.array(a_slice)

        info = model.encode({"pixels": p_tensor, "action": a_tensor})
        gt_emb = info["emb"][0]       # [T, D]
        act_emb = info["act_emb"][0]  # [T, D]

        # Context frames
        z_curr = gt_emb[:history_size][np.newaxis, ...]   # [1, H, D]
        a_curr = act_emb[:history_size][np.newaxis, ...]  # [1, H, D]

        for k in range(1, max_horizon + 1):
            pred_step = model.predict(z_curr, a_curr)[:, -1:, :]  # [1, 1, D]
            tgt_step = gt_emb[history_size + k - 1 : history_size + k][np.newaxis, ...]  # [1, 1, D]

            diff_sq = mx.mean(mx.square(pred_step - tgt_step)).item()
            p_n = mx.linalg.norm(pred_step, axis=-1, keepdims=True)
            t_n = mx.linalg.norm(tgt_step, axis=-1, keepdims=True)
            cos = (
                mx.sum(pred_step * tgt_step, axis=-1, keepdims=True)
                / (p_n * t_n + 1e-8)
            ).item()

            step_errors[k].append(diff_sq)
            step_cosines[k].append(cos)

            # Autoregressive context update
            z_curr = mx.concatenate([z_curr[:, 1:, :], pred_step], axis=1)
            next_a = act_emb[history_size + k - 1 : history_size + k][np.newaxis, ...]
            a_curr = mx.concatenate([a_curr[:, 1:, :], next_a], axis=1)

    results: Dict[int, Dict[str, float]] = {}
    for k in range(1, max_horizon + 1):
        results[k] = {
            "mse": float(np.mean(step_errors[k])) if step_errors[k] else float("nan"),
            "cos_sim": float(np.mean(step_cosines[k])) if step_cosines[k] else float("nan"),
        }
    return results


def evaluate_planning(
    model: JEPA,
    dataset: PushTMiniDataset,
    eval_episodes: List[int],
    history_size: int = 3,
    horizon: int = 5,
    num_samples: int = 256,
    num_elites: int = 25,
    iterations: int = 8,
) -> Dict[str, float]:
    """Evaluates CEM goal-directed planning over a fixed horizon."""
    total_horizon = history_size + horizon
    planner = CEMPlanner(
        model=model,
        planning_horizon=total_horizon,
        action_dim=10,
        num_samples=num_samples,
        num_elites=num_elites,
        iterations=iterations,
        alpha=0.10,
    )
    shooting_planner = ShootingPlanner(
        model=model,
        planning_horizon=total_horizon,
        action_dim=10,
        num_samples=num_samples,
    )

    c_init_list, c_final_list, c_shoot_list = [], [], []

    for ep_idx in eval_episodes:
        ep_pixels = dataset.episodes_pixels[ep_idx]
        seq_len = total_horizon + 2
        if len(ep_pixels) < seq_len:
            continue

        ctx = (ep_pixels[:history_size].astype(np.float32) / 255.0)[np.newaxis, ...]
        goal = (ep_pixels[total_horizon : total_horizon + 1].astype(np.float32) / 255.0)[np.newaxis, ...]

        ctx_norm = imagenet_normalize(mx.array(ctx))
        goal_norm = imagenet_normalize(mx.array(goal))

        info = {
            "pixels": mx.expand_dims(ctx_norm, 1),
            "goal": mx.expand_dims(goal_norm, 1),
        }

        _, shoot_cost, _ = shooting_planner.plan(info)
        c_shoot_list.append(float(shoot_cost))

        _, best_cost, history = planner.plan(info)
        c_init_list.append(history[0])
        c_final_list.append(float(best_cost))

    init_c = float(np.mean(c_init_list))
    final_c = float(np.mean(c_final_list))
    shoot_c = float(np.mean(c_shoot_list))
    reduc_pct = float((init_c - final_c) / (init_c + 1e-8) * 100.0)
    margin_pct = float((shoot_c - final_c) / (shoot_c + 1e-8) * 100.0)

    return {
        "initial_cost": init_c,
        "final_cost": final_c,
        "shooting_cost": shoot_c,
        "cost_reduction_pct": reduc_pct,
        "margin_over_shooting_pct": margin_pct,
    }


def main() -> None:
    print("Loading PushTMiniDataset for horizon and planning evaluation...")
    dataset = PushTMiniDataset(num_episodes=100, frameskip=5, img_size=96)
    eval_episodes = list(range(70, 100))  # Held-out 30 test episodes

    models = {
        "k1": Path("outputs/experiments/lewm_k1.npz"),
        "k2": Path("outputs/experiments/lewm_k2.npz"),
        "k3": Path("outputs/experiments/lewm_k3.npz"),
        "ar_k3": Path("outputs/experiments/lewm_ar_k3.npz"),
        "comp_k3": Path("outputs/experiments/lewm_comp_k3.npz"),
    }

    rollout_results: Dict[str, Dict[int, Dict[str, float]]] = {}
    planning_results: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 80)
    print("PART 1: MULTI-STEP AUTOREGRESSIVE ROLLOUT ACCURACY (1 to 8 STEPS)")
    print("=" * 80)

    for name, path in models.items():
        if not path.exists():
            continue
        print(f"Evaluating model: {name}...")
        model = build_model(str(path))
        rollout_results[name] = evaluate_rollout_horizons(
            model=model,
            dataset=dataset,
            eval_episodes=eval_episodes,
            history_size=3,
            max_horizon=8,
        )

    # Print rollout table
    print("\n--- Rollout Cosine Similarity across Horizons (Higher is Better) ---")
    header = f"{'Model':<10} | " + " | ".join([f"k={k}" for k in range(1, 9)])
    print(header)
    print("-" * len(header))
    for name, res in rollout_results.items():
        row = f"{name:<10} | " + " | ".join([f"{res[k]['cos_sim']:+.3f}" for k in range(1, 9)])
        print(row)

    print("\n--- Rollout MSE across Horizons (Lower is Better) ---")
    header_mse = f"{'Model':<10} | " + " | ".join([f"k={k}" for k in range(1, 9)])
    print(header_mse)
    print("-" * len(header_mse))
    for name, res in rollout_results.items():
        row = f"{name:<10} | " + " | ".join([f"{res[k]['mse']:.4f}" for k in range(1, 9)])
        print(row)

    print("\n" + "=" * 80)
    print("PART 2: CEM GOAL PLANNING EVALUATION (Horizon=5, Samples=256, Iters=8)")
    print("=" * 80)

    for name, path in models.items():
        if not path.exists():
            continue
        print(f"Running CEM Planning on held-out episodes for {name}...")
        model = build_model(str(path))
        p_res = evaluate_planning(
            model=model,
            dataset=dataset,
            eval_episodes=eval_episodes[:15],  # 15 held-out episodes
            history_size=3,
            horizon=5,
        )
        planning_results[name] = p_res

    print("\n--- Planning Performance Summary ---")
    print(f"{'Model':<10} | {'Init Cost':<10} | {'Shooting':<10} | {'CEM Final':<10} | {'Reduction %':<12} | {'Margin over Shoot %':<20}")
    print("-" * 80)
    for name, p_res in planning_results.items():
        print(
            f"{name:<10} | {p_res['initial_cost']:<10.2f} | {p_res['shooting_cost']:<10.2f} | "
            f"{p_res['final_cost']:<10.2f} | {p_res['cost_reduction_pct']:<12.1f}% | "
            f"{p_res['margin_over_shooting_pct']:<20.1f}%"
        )

    # Save results to JSON
    out_file = Path("outputs/experiments/horizon_and_planning_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "rollouts": rollout_results,
                "planning": planning_results,
            },
            f,
            indent=2,
        )
    print(f"\nAll results saved to {out_file}")


if __name__ == "__main__":
    main()
