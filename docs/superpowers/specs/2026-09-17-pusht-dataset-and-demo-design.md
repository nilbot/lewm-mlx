# Design Document: PushT Mini-Dataset and Dual-Mode Inference Demo

## 1. Context and Motivation
The `lewm-mlx` repository provides a native Apple Silicon MLX port of the LeWorldModel (LeWM) Joint-Embedding Predictive Architecture (JEPA). While core architecture components and synthetic training loops are verified via unit tests, the repository lacked:
1. A reproducible, lightweight toy dataset reflecting the benchmark robotics environment evaluated in the original LeWM paper (Push-T).
2. Dataloader batching and frameskip action formatting integrated with `train.py`.
3. A vectorised goal-directed planner (Cross-Entropy Method) and multi-step autoregressive rollout inference demo (`demo.py`).

This document specifies the technical design for the dataset pipeline, training loop integration, and inference demo CLI.

---

## 2. Architecture & Component Decomposition

### 2.1 Component 1: `PushTMiniDataset` (`lewm_mlx/dataset.py`)
* **Purpose**: Download, extract, process, cache, and sample PushT demonstration trajectories.
* **Source**:
  - Parquet metadata & actions: `lerobot/pusht` on Hugging Face (`data/chunk-000/file-000.parquet`, 674 KB).
  - Observation video: `lerobot/pusht` (`videos/observation.image/chunk-000/file-000.mp4`, 6.89 MB).
* **Local Storage & Cache**:
  - Compressed `.npz` archive at `~/.cache/lewm/pusht_mini_10ep.npz` (~15 MB).
  - First invocation downloads chunk-0, extracts `num_episodes` (default: 10), resizes frames, normalizes actions, and writes the cache file. Subsequent calls load from disk in `< 50 ms`.
* **Action & Observation Chunking**:
  - Actions: 2D continuous end-effector coordinates $(x, y)$ normalized to $[-1, 1]$.
  - Frameskip $F$ (default: 5): consecutive actions are concatenated into macro-actions of dimension $D_{act} = F \times 2 = 10$.
  - Sampling API: `sample_batch(batch_size, history_size, num_preds)` returning:
    - `"pixels"`: MLX array of shape $[B, H + K, 3, \text{img\_size}, \text{img\_size}]$ in $[0, 1]$.
    - `"action"`: MLX array of shape $[B, H + K, D_{act}]$.
* **Fault Tolerance**:
  - If Hugging Face Hub is unreachable (e.g. offline execution), falls back to a synthetic kinematics generator with a warning.

### 2.2 Component 2: Training Loop Integration (`lewm_mlx/train.py`)
* **CLI Parameterization**:
  - `--dataset`: `pusht_mini` (default) or `synthetic`.
  - `--frameskip`: Action frame skip factor (default: 5).
  - `--img-size`: Frame resolution (default: 96, with support for 224).
* **Model Dynamic Configuration**:
  - Automatically matches `action_encoder = Embedder(input_dim = frameskip * 2, ...)`.
  - Seamlessly handles batches from `PushTMiniDataset` or `generate_synthetic_batch`.

### 2.3 Component 3: Vectorised CEM Planner (`lewm_mlx/planner.py`)
* **Purpose**: Solve goal-directed trajectory optimization using the learned world model in latent space.
* **Mathematical Formulation**:
  Let the planning horizon be $T_{plan}$ and candidate population size be $S$.
  Given initial state history context $s_{0:H}$ and target goal image $g$:
  1. Initialize distribution parameters: $\boldsymbol{\mu}_0 = \mathbf{0} \in \mathbb{R}^{T_{plan} \times D_{act}}$, $\boldsymbol{\sigma}_0 = 0.5 \cdot \mathbf{I}$.
  2. For iteration $k = 1, \dots, N_{iter}$:
     - Sample $S$ action trajectories:
       $$\mathbf{A}^{(s)} \sim \text{clip}\left(\mathcal{N}(\boldsymbol{\mu}_{k-1}, \boldsymbol{\sigma}_{k-1}^2), -1.0, 1.0\right), \quad s \in \{1, \dots, S\}$$
     - Evaluate step cost using JEPA latent distance:
       $$\mathcal{J}(\mathbf{A}^{(s)}) = \text{model.get\_cost}(\text{info}, \mathbf{A}^{(s)})$$
     - Select elite subset $\mathcal{E} \subset \{1, \dots, S\}$ corresponding to the top $N_{elite}$ lowest costs.
     - Update distribution with momentum parameter $\alpha \in [0, 1]$:
       $$\boldsymbol{\mu}_k = \alpha \boldsymbol{\mu}_{k-1} + (1 - \alpha) \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \mathbf{A}^{(e)}$$
       $$\boldsymbol{\sigma}_k = \alpha \boldsymbol{\sigma}_{k-1} + (1 - \alpha) \sqrt{\frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} (\mathbf{A}^{(e)} - \boldsymbol{\mu}_k)^2 + \epsilon}$$
  3. Output the optimal planned action sequence $\mathbf{A}^* = \boldsymbol{\mu}_{N_{iter}}$ and minimal cost.

### 2.4 Component 4: Unified Demo CLI (`demo.py`)
* **Purpose**: Executable script for demonstrating trained or restored JEPA model inference.
* **Modes**:
  - `--mode rollout`: Given an episode from `PushTMiniDataset`, encode context frames $T_{0:H}$, apply ground-truth actions, and autoregressively predict next latents. Compute step-by-step MSE error between predicted latents and true target embeddings:
    $$\text{MSE}_t = \frac{1}{D} \sum_{d=1}^D (\hat{z}_{t, d} - z_{t, d})^2$$
  - `--mode plan`: Execute `CEMPlanner` to find actions reaching a distant goal image from the dataset, verifying optimization over random shooting baseline.
  - `--save-plot`: Save a 3-panel figure visualizing context frames, goal frame, and rollout/planning convergence curves.

---

## 3. Data Flow & Interfaces

```
[PushT Video (96x96 RGB) + Parquet Actions]
                   │
                   ▼ (Downsample / Normalize)
        [PushTMiniDataset Cache]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   [train.py Batch]     [demo.py Initial History + Goal]
         │                   │
         ▼                   ▼
    [JEPA.encode]       [JEPA.encode (Single Pass)]
         │                   │
         ▼                   ▼
  [JEPA.predict]        [CEMPlanner: S Candidates]
         │                   │
         ▼                   ▼
   [Loss & Update]      [JEPA.get_cost (Latent MSE)]
                             │
                             ▼
                        [Optimal Plan A* & Artifact Plot]
```

---

## 4. Verification & Testing Strategy

1. **Unit Tests**:
   - `tests/test_dataset.py`:
     - Test dataset caching logic and file layout.
     - Verify batch dimensions: `pixels` $[B, T, 3, H, W]$, `actions` $[B, T, D_{act}]$.
     - Check absence of NaNs, Infs, and range compliance ($[-1, 1]$ for actions, $[0, 1]$ for pixels).
   - `tests/test_planner.py`:
     - Test `CEMPlanner` with mock/tiny JEPA model.
     - Verify candidate vectorization and cost reduction across iterations.
2. **Regression & Integration Tests**:
   - Existing 6 unit tests in `tests/test_comparison.py` must continue to pass.
   - `lewm_mlx/train.py --dataset pusht_mini --epochs 1 --steps-per-epoch 3` must complete and generate `lewm_weights.npz`.
   - `python demo.py --mode both --save-plot pusht_demo.png` must execute and output the verification figure.
