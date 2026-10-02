r"""Generates diagnostic research figures summarizing Le World Model experiments.

Produces a 3-panel publication-grade figure:
- Panel 1: Multi-Step Rollout Error (k=1..8) across architectures.
- Panel 2: Latent Distance vs. Physical Euclidean Distance correlation.
- Panel 3: CEM Planning Optimization Curves over iterations.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from lewm_mlx.dataset import PushTMiniDataset
from lewm_mlx.preprocessing import imagenet_normalize
import mlx.core as mx
from scripts.probe_latent_state import build_model, load_aligned_states


def main() -> None:
    print("Generating comprehensive research diagnostic figures...")
    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "research_insights.png"

    # Load benchmark results
    bench_file = out_dir / "horizon_and_planning_benchmark.json"
    with open(bench_file, "r") as f:
        bench_data = json.load(f)

    rollouts = bench_data["rollouts"]

    # Load dataset and states for metric correlation
    dataset = PushTMiniDataset(num_episodes=40, frameskip=5, img_size=96)
    states = load_aligned_states(num_episodes=40, frameskip=5)
    all_states = np.concatenate(states, axis=0)

    np.random.seed(42)
    N = len(all_states)
    idx1 = np.random.randint(0, N, size=1500)
    idx2 = np.random.randint(0, N, size=1500)
    phys_dist = np.linalg.norm(all_states[idx1] - all_states[idx2], axis=1)

    # Encode frames with k2 model
    m_k2 = build_model("outputs/experiments/lewm_k2.npz")
    all_embs = []
    for ep in range(40):
        ep_p = dataset.episodes_pixels[ep]
        p_norm = (ep_p.astype(np.float32) / 255.0)[np.newaxis, ...]
        p_tensor = imagenet_normalize(mx.array(p_norm))
        info = m_k2.encode({"pixels": p_tensor})
        all_embs.append(np.array(info["emb"][0]))
    all_embs = np.concatenate(all_embs, axis=0)
    latent_dist = np.linalg.norm(all_embs[idx1] - all_embs[idx2], axis=1)
    r_val, _ = pearsonr(latent_dist, phys_dist)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)

    # -------------------------------------------------------------
    # Panel 1: Multi-Step Rollout MSE
    # -------------------------------------------------------------
    ax1 = axes[0]
    horizons = list(range(1, 9))
    colors = {
        "k1": "#1f77b4",
        "k2": "#2ca02c",
        "k3": "#ff7f0e",
        "ar_k3": "#d62728",
        "comp_k3": "#9467bd",
    }
    labels = {
        "k1": "K=1 (1-step jump)",
        "k2": "K=2 (2-step jump)",
        "k3": "K=3 (3-step jump)",
        "ar_k3": "K=3 Autoregressive BPTT",
        "comp_k3": "K=3 Teacher-Forced",
    }

    for name in ["k1", "k2", "k3", "ar_k3", "comp_k3"]:
        if name in rollouts:
            mses = [rollouts[name][str(k)]["mse"] for k in horizons]
            ax1.plot(horizons, mses, marker="o", label=labels.get(name, name), color=colors.get(name))

    ax1.set_title("Autoregressive Rollout Error vs. Horizon (k=1..8)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rollout Step Horizon (k)", fontsize=11)
    ax1.set_ylabel("Latent Rollout MSE", fontsize=11)
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Latent Distance vs Physical Distance Metric Grounding
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.scatter(phys_dist, latent_dist, alpha=0.25, s=12, color="#2ca02c", edgecolors="none")
    # Linear fit line
    m_slope, b_intercept = np.polyfit(phys_dist, latent_dist, 1)
    x_line = np.linspace(phys_dist.min(), phys_dist.max(), 100)
    ax2.plot(x_line, m_slope * x_line + b_intercept, color="darkgreen", lw=2, label=f"Fit (r = +{r_val:.3f})")

    ax2.set_title("Latent Space Metric Preservation (K=2)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Physical Euclidean Distance (pixels)", fontsize=11)
    ax2.set_ylabel("Latent L2 Distance ||z1 - z2||", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # -------------------------------------------------------------
    # Panel 3: CEM Planning Cost Reduction across Models
    # -------------------------------------------------------------
    ax3 = axes[2]
    # Bar chart comparing cost reduction and margin over shooting
    models_list = ["k1", "k2", "k3", "ar_k3", "comp_k3"]
    plan_data = bench_data["planning"]
    reductions = [plan_data[m]["cost_reduction_pct"] for m in models_list]
    margins = [plan_data[m]["margin_over_shooting_pct"] for m in models_list]

    x = np.arange(len(models_list))
    width = 0.35

    ax3.bar(x - width/2, reductions, width, label="CEM Cost Reduction %", color="#1f77b4", alpha=0.85)
    ax3.bar(x + width/2, margins, width, label="Margin over Shooting %", color="#ff7f0e", alpha=0.85)

    ax3.set_title("5-Step Goal Planning Convergence (S=256, I=8)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Percentage (%)", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels([labels[m].replace(" ", "\n") for m in models_list], fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Research diagnostic figure successfully saved to: {out_plot}")


if __name__ == "__main__":
    main()
