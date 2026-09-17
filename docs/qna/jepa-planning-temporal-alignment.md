# How Are Context History, Prediction Horizon, and Target Goal Frames Aligned in JEPA Rollouts?

## Context
In the Joint-Embedding Predictive Architecture (JEPA), the world model predicts future latent states conditioned on past visual observations and continuous action trajectories. During the implementation of the rollout and planning pipeline (`lewm_mlx/jepa.py`, `lewm_mlx/planner.py`, and `demo.py`), subtle off-by-one errors frequently arise when indexing future latent predictions to evaluate goal reachability costs.

## Answer

### 1. Trajectory Temporal Decomposition
A trajectory segment is defined by two fundamental temporal parameters:
- $H$: **Context History Length** (number of observed initial visual frames, typically $H = 3$).
- $K$: **Prediction / Planning Horizon** (number of future steps the model predicts, typically $K = 5$).
- $T = H + K$: **Total Temporal Horizon** ($T = 8$).

The temporal indices are structured as follows:

$$\underbrace{s_0, s_1, \dots, s_{H-1}}_{\text{Context Frames (Observed)}} \quad \longrightarrow \quad \underbrace{\hat{z}_H, \hat{z}_{H+1}, \dots, \hat{z}_{H+K-1}}_{\text{Latent Rollout Predictions (Autoregressive)}}$$

1. **Context Encoding**: The target encoder $\text{ViT}$ processes observed frames $s_{0:H}$ yielding deterministic latent representations $z_{0:H-1}$ of shape $[B, H, D_{emb}]$.
2. **Autoregressive Latent Rollout**: Conditioned on $z_{0:H-1}$ and continuous action embeddings $a_{0:T-1}$, the AR predictor unrolls latent representations step-by-step for $t = H, H+1, \dots, H+K-1$.

### 2. Goal Alignment and Slicing Pitfall
When executing goal-directed trajectory optimization (via random shooting or CEM), the planner aims to steer the system to match a target goal observation $s_{goal}$.

In Python zero-indexed notation:
- Ground-truth context states occupy indices `0` through `H - 1`.
- The first predicted future state $\hat{z}_H$ is located at index `H`.
- The $K$-th predicted future state $\hat{z}_{H+K-1}$ is located at index `H + K - 1`.

```python
# Context observation: [B, 1, H, 3, img_size, img_size]
# Goal observation:    [B, 1, 1, 3, img_size, img_size]
context_pixels = batch["pixels"][:, :, :history_size]   # frames 0..H-1
goal_pixels = batch["pixels"][:, :, history_size + horizon - 1: history_size + horizon] # frame H+K-1
```

A common bug is setting the target frame to `history_size + horizon` (i.e. $H + K$), which either attempts an out-of-bounds slice on a batch of length $T = H + K$ or compares against a state one timestep beyond the actual planned action sequence.

### 3. Latent Cost Formulation
The goal-reaching objective evaluates the Euclidean squared error between the projected prediction at the terminal horizon and the projected target goal:

$$J(\mathbf{A}) = \| \text{pred\_proj}(\hat{z}_{H+K-1}) - \text{projector}(z_{goal}) \|_2^2$$

In `JEPA.get_cost`, the cost is computed efficiently across all $S$ candidates in parallel:
```python
# rollout_latents shape: [B, S, T, D_emb]
pred_terminal = rollout_latents[:, :, -1, :]  # Terminal predicted step [B, S, D_emb]

# Embed goal: [B, 1, D_emb]
goal_emb = self.projector(self.encode(goal_pixels))

# Squared L2 loss across latent dimension: [B, S]
costs = mx.sum(mx.square(pred_terminal - goal_emb), axis=-1)
```
Using the terminal index `-1` on `rollout_latents` guarantees exact temporal alignment with the end of the specified planning horizon $T$.
