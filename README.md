# Le World Model in MLX (`lewm-mlx`)

A native Apple Silicon ([MLX](https://github.com/ml-explore/mlx)) implementation of the **LeWorldModel (LeWM)** Joint-Embedding Predictive Architecture (JEPA) for robotics and visual world modeling.

This repository provides an end-to-end research and evaluation pipeline optimized for unified memory on macOS, including a Vision Transformer (ViT) patch encoder, autoregressive latent transition model, SIGReg (Skew-Information-Geometric) latent regularization, vectorized Cross-Entropy Method (CEM) planning, and demonstration dataset loaders.

---

## Key Features

- **Native MLX Core**: Complete re-implementation of the PyTorch reference using pure MLX primitives (`mlx.core`, `mlx.nn`), fully compatible with unified Apple Silicon memory architectures.
- **Push-T Demonstration Dataset Pipeline**: Built-in downloader, action normalizer, and compressed local cache for the standard Push-T benchmark (`lerobot/pusht` on Hugging Face), featuring automatic offline synthetic fallback.
- **Autoregressive Latent Dynamics**: Multi-step autoregressive rollouts conditioned on continuous chunked action sequences in latent feature space.
- **Vectorized Trajectory Planners**: Parallelized Cross-Entropy Method (CEM) and Random Shooting trajectory optimizers implemented directly in MLX for goal-directed latent control.
- **Multi-Panel Diagnostic Visualizer**: Unified demo CLI generating multi-panel visual diagnostic artifacts showing context frames, goal observations, rollout MSE trajectories, and optimization convergence curves.

---

## Installation

Ensure you have Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) installed on macOS Apple Silicon.

```bash
git clone https://github.com/nilbot/lewm-mlx.git
cd lewm-mlx
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Dataset: Push-T Mini Pipeline

The dataset loader is implemented in [`lewm_mlx/dataset.py`](lewm_mlx/dataset.py) (`PushTMiniDataset`).

It downloads and processes demonstration trajectories directly from the Hugging Face Hub repository (`lerobot/pusht`):
1. **Metadata & Video**: Fetches the initial chunk (`file-000.parquet` and `file-000.mp4`).
2. **Action Normalization**: Rescales raw continuous coordinate actions $[0, 512]$ to $[-1.0, 1.0]$.
3. **Macro-Action Chunking**: Groups $F$ consecutive actions into single macro-actions of dimension $D_{act} = F \times 2$ (default frameskip $F = 5 \implies D_{act} = 10$).
4. **Local Caching**: Stores extracted episodes in a compressed `.npz` archive at `~/.cache/lewm/pusht_mini_<num_episodes>ep.npz` for subsequent instant loads ($< 50\text{ ms}$).
5. **Offline Fallback**: Automatically activates a synthetic kinematics trajectory generator if the Hugging Face Hub is unreachable.

---

## Training on Push-T Mini

Train the world model directly on Push-T demonstration episodes using the integrated training script:

```bash
PYTHONPATH=. uv run python lewm_mlx/train.py \
    --dataset pusht_mini \
    --num-episodes 5 \
    --epochs 2 \
    --steps-per-epoch 5 \
    --img-size 96 \
    --save-path lewm_weights.npz
```

To train on synthetic kinematics batches instead:

```bash
PYTHONPATH=. uv run python lewm_mlx/train.py \
    --dataset synthetic \
    --epochs 2 \
    --steps-per-epoch 5
```

---

## Inference and Planning Demo CLI

The repository includes a unified evaluation CLI [`demo.py`](demo.py) supporting both autoregressive rollout evaluation and goal-directed CEM planning.

### Running Both Modes (Recommended)

```bash
PYTHONPATH=. uv run python demo.py \
    --weights lewm_weights.npz \
    --mode both \
    --num-episodes 5 \
    --img-size 96 \
    --save-plot pusht_demo.png
```

### Running Specific Modes

- **Latent Rollout Only**:
  ```bash
  PYTHONPATH=. uv run python demo.py --mode rollout --num-episodes 5 --img-size 96 --save-plot rollout.png
  ```
- **Goal-Directed Planning Only**:
  ```bash
  PYTHONPATH=. uv run python demo.py --mode plan --num-episodes 5 --img-size 96 --save-plot plan.png
  ```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--weights` | `str` | `lewm_weights.npz` | Path to saved model weights archive (`.npz`). |
| `--mode` | `str` | `both` | Evaluation mode (`rollout`, `plan`, or `both`). |
| `--num-episodes` | `int` | `10` | Number of demonstration episodes to load. |
| `--frameskip` | `int` | `5` | Action frameskip factor ($D_{act} = 2 \times \text{frameskip}$). |
| `--img-size` | `int` | `96` | Frame resolution (e.g. 96 or 224). |
| `--history-size` | `int` | `3` | Context observation history length ($H$). |
| `--horizon` | `int` | `5` | Forward prediction / planning horizon ($K$). |
| `--save-plot` | `str` | `pusht_demo.png` | Destination path for diagnostic plot. |
| `--cache-dir` | `str` | `None` | Local directory for cached dataset archives. |

---

## Running Automated Tests

Run the complete test suite across architectural equivalence, dataset caching, planners, and integration tests:

```bash
PYTHONPATH=. uv run pytest tests/ -v
```

All 28 tests across the repository verify numerical correctness, gradient propagation, and interface parity against the PyTorch reference implementation.

---

## Documentation & Architecture

For further architectural and mathematical details, refer to:
- [Push-T Dataset & Demo Design Document](docs/design/pusht_dataset_and_demo.md)
- [Push-T Demo Execution Journal](docs/journal/2026-09-17-pusht-demo.md)
