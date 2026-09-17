"""Training pipeline for Le World Model in MLX.

Trains a Joint-Embedding Predictive Architecture (JEPA) on visual demonstration
trajectories from Push-T (PushTMiniDataset) or procedurally generated synthetic
trajectories using Self-Information-Gauged Regularization (SIGReg) and latent prediction loss.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

from lewm_mlx.dataset import PushTMiniDataset
from lewm_mlx.jepa import JEPA
from lewm_mlx.module import ARPredictor, Embedder, MLP, SIGReg
from lewm_mlx.vit import ViTModel


def compute_tree_norm(tree: Any) -> mx.array:
    r"""Computes Euclidean $L_2$ norm over a nested parameter or gradient tree.

    $$\|\mathbf{T}\|_2 = \sqrt{\sum_{i} \|\mathbf{T}_i\|_2^2}$$

    Args:
        tree: Nested dictionary, list, or array structure containing MLX arrays.

    Returns:
        Scalar MLX array representing the aggregate Euclidean norm.
    """
    flat = tree_flatten(tree)
    if not flat:
        return mx.array(0.0)
    return mx.sqrt(sum(mx.sum(mx.square(p)) for _, p in flat))


def generate_synthetic_batch(
    batch_size: int, seq_len: int, img_size: int, action_dim: int
) -> Dict[str, mx.array]:
    r"""Generates a synthetic batch of visual and action trajectories for testing.

    Args:
        batch_size: Number of sequences in batch ($B$).
        seq_len: Temporal length of trajectory sequences ($T = H + K$).
        img_size: Square image spatial resolution ($H = W$).
        action_dim: Dimension of the action vector ($D_{\\text{act}}$).

    Returns:
        Dict containing:
            "pixels": MLX array of shape $[B, T, 3, \\text{img\\_size}, \\text{img\\_size}]$.
            "action": MLX array of shape $[B, T, \\text{action\\_dim}]$.
    """
    # pixels: [B, T, 3, img_size, img_size]
    pixels = mx.random.normal(shape=(batch_size, seq_len, 3, img_size, img_size))
    # actions: [B, T, action_dim]
    actions = mx.random.normal(shape=(batch_size, seq_len, action_dim))
    # Simulate occasional missing boundaries with NaNs
    nan_mask = mx.random.uniform(shape=(batch_size, seq_len, 1)) < 0.05
    actions = mx.where(nan_mask, mx.array(float("nan")), actions)

    return {
        "pixels": pixels,
        "action": actions,
    }


def main() -> None:
    """Parses arguments and runs the training loop for JEPA in MLX."""
    parser = argparse.ArgumentParser(description="Train Le World Model in MLX")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pusht_mini", "synthetic"],
        default="pusht_mini",
        help="Dataset to train on (pusht_mini or synthetic)",
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
        "--cache-dir",
        type=str,
        default=None,
        help="Local directory for dataset cache archive",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=96, help="Input image size")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--history-size", type=int, default=3, help="History context size"
    )
    parser.add_argument(
        "--num-preds", type=int, default=1, help="Number of steps to predict"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=20, help="Steps per epoch"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument(
        "--sigreg-weight", type=float, default=0.09, help="SIGReg loss weight"
    )
    parser.add_argument(
        "--save-path", type=str, default="lewm_weights.npz", help="Path to save weights"
    )
    parser.add_argument(
        "--grad-breakdown",
        action="store_true",
        help="Display per-module gradient norms (encoder, predictor, action_encoder, heads)",
    )
    parser.add_argument(
        "--eval-fixed",
        action="store_true",
        help="Periodically evaluate and log metrics on a fixed held-out probe batch",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional destination path for structured JSON Lines (.jsonl) telemetry",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Log per-step telemetry every N steps within each epoch (0 disables per-step logging)",
    )
    args = parser.parse_args()

    # Configure action dimension depending on dataset choice:
    # For pusht_mini, action_dim = frameskip * 2 (2D end-effector coordinates chunked over frameskip).
    # For synthetic, action_dim = history_size * 2.
    action_dim = (
        args.frameskip * 2 if args.dataset == "pusht_mini" else args.history_size * 2
    )

    # Instantiate dataset if pusht_mini requested
    dataset = None
    if args.dataset == "pusht_mini":
        dataset = PushTMiniDataset(
            cache_dir=args.cache_dir,
            num_episodes=args.num_episodes,
            frameskip=args.frameskip,
            img_size=args.img_size,
        )

    # Define model sub-components
    # ViT visual encoder for patch-based representation
    encoder = ViTModel(
        image_size=args.img_size,
        patch_size=16,
        num_channels=3,
        hidden_size=args.embed_dim,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
    )

    # Autoregressive transformer predictor in latent space
    predictor = ARPredictor(
        num_frames=args.history_size,
        depth=2,
        heads=2,
        mlp_dim=256,
        input_dim=args.embed_dim,
        hidden_dim=args.embed_dim,
    )

    # Action encoder projecting continuous control chunks to latent embedding space
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=args.embed_dim,
        emb_dim=args.embed_dim,
    )

    # Projection heads
    projector = MLP(
        input_dim=args.embed_dim,
        hidden_dim=256,
        output_dim=args.embed_dim,
        norm_fn=nn.LayerNorm,
    )

    pred_proj = MLP(
        input_dim=args.embed_dim,
        hidden_dim=256,
        output_dim=args.embed_dim,
        norm_fn=nn.LayerNorm,
    )

    # Joint-Embedding Predictive Architecture (JEPA)
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.train()

    # Self-Information-Gauged Regularization (SIGReg)
    sigreg = SIGReg(knots=17, num_proj=256)

    # Optimizer
    optimizer = opt.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Sequence length: T = H + K (history_size + num_preds)
    seq_len = args.history_size + args.num_preds

    # Define loss function
    def loss_fn(
        model: JEPA, batch: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        r"""Computes combined latent prediction and SIGReg regularization loss alongside diagnostics.

        Loss formulation:
            $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{SIGReg}} \mathcal{L}_{\text{SIGReg}}$$
            $$\mathcal{L}_{\text{pred}} = \frac{1}{B \cdot H \cdot D} \| \hat{\mathbf{s}}_{1:H} - \mathbf{s}_{K:H+K} \|_2^2$$

        Args:
            model: JEPA model instance.
            batch: Dictionary containing "pixels" and "action" MLX tensors:
                - pixels: $[B, T, 3, H, W]$
                - action: $[B, T, D_{\text{act}}]$

        Returns:
            Tuple of (total_loss, metrics):
                total_loss: Optimization scalar objective $\mathcal{L}_{\text{total}}$.
                metrics: Dictionary mapping metric identifiers to scalar diagnostic tensors.
        """
        # actions: [B, T, D_act]
        actions = batch["action"]
        actions = mx.where(mx.isnan(actions), mx.array(0.0), actions)

        batch_new = {
            "pixels": batch["pixels"],
            "action": actions,
        }

        output = model.encode(batch_new)
        emb = output["emb"]  # [B, T, D]
        act_emb = output["act_emb"]  # [B, T, D]

        # Extract context embeddings: [B, H, D]
        ctx_emb = emb[:, : args.history_size, :]
        ctx_act = act_emb[:, : args.history_size, :]

        # Target future embeddings: [B, H, D]
        tgt_emb = emb[:, args.num_preds :, :]
        # Predictor forward pass: [B, H, D]
        pred_emb = model.predict(ctx_emb, ctx_act)

        # Latent prediction mean-squared error
        pred_loss = mx.mean(mx.square(pred_emb - tgt_emb))
        # SIGReg expects [T, B, D] for marginal regularization over batch
        sigreg_loss = sigreg(emb.transpose(1, 0, 2))
        total_loss = pred_loss + args.sigreg_weight * sigreg_loss

        # Latent space geometry and collapse diagnostics:
        pred_norm = mx.linalg.norm(pred_emb, axis=-1, keepdims=True)  # [B, H, 1]
        tgt_norm = mx.linalg.norm(tgt_emb, axis=-1, keepdims=True)    # [B, H, 1]
        # Cosine similarity between predictions and targets: scalar
        cos_sim = mx.mean(
            mx.sum(pred_emb * tgt_emb, axis=-1, keepdims=True)
            / (pred_norm * tgt_norm + 1e-8)
        )
        # Batch dispersion across channel dimensions (detects dimensional or representation collapse)
        emb_std = mx.mean(mx.std(emb, axis=0))
        # Embedding and prediction L2 energy
        emb_norm = mx.mean(mx.linalg.norm(emb, axis=-1))
        pred_l2_norm = mx.mean(pred_norm)

        metrics = {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "cos_sim": cos_sim,
            "emb_std": emb_std,
            "emb_norm": emb_norm,
            "pred_norm": pred_l2_norm,
        }
        return total_loss, metrics

    # Step function compiled for MLX GPU/Metal execution
    loss_and_grads = nn.value_and_grad(model, loss_fn)

    def step_fn(
        batch: Dict[str, mx.array],
    ) -> Dict[str, mx.array]:
        r"""Executes a single forward, backward, optimizer, and telemetry step.

        Args:
            batch: Dictionary mapping feature keys ("pixels", "action") to MLX arrays.

        Returns:
            Dictionary containing loss components, representation metrics, and gradient norms.
        """
        (loss, metrics), grads = loss_and_grads(model, batch)
        optimizer.update(model, grads)

        # Diagnostic gradient and parameter dynamics
        metrics["grad_norm"] = compute_tree_norm(grads)
        metrics["enc_grad_norm"] = compute_tree_norm(grads.get("encoder", {}))
        metrics["pred_grad_norm"] = compute_tree_norm(grads.get("predictor", {}))
        metrics["act_grad_norm"] = compute_tree_norm(grads.get("action_encoder", {}))
        metrics["proj_grad_norm"] = compute_tree_norm(grads.get("projector", {}))
        metrics["pred_proj_grad_norm"] = compute_tree_norm(grads.get("pred_proj", {}))
        metrics["param_norm"] = compute_tree_norm(model.parameters())
        return metrics

    state = [model.state, optimizer.state]
    train_step = mx.compile(step_fn, inputs=state, outputs=state)

    # Optional fixed evaluation probe to disentangle stochastic batch sampling noise
    eval_batch = None
    if args.eval_fixed:
        if dataset is not None:
            eval_batch = dataset.sample_batch(
                args.batch_size, args.history_size, args.num_preds
            )
        else:
            eval_batch = generate_synthetic_batch(
                args.batch_size, seq_len, args.img_size, action_dim
            )

    # Reset metrics file if specified
    if args.metrics_path:
        metrics_p = Path(args.metrics_path)
        metrics_p.parent.mkdir(parents=True, exist_ok=True)
        if metrics_p.exists():
            metrics_p.unlink()

    print("Starting MLX Le World Model Training Loop...")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Steps per epoch: {args.steps_per_epoch}")
    print(f"Batch size: {args.batch_size}, Image size: {args.img_size}")
    if args.metrics_path:
        print(f"Metrics log: {args.metrics_path}")
    if args.eval_fixed:
        print("Fixed evaluation probe: enabled")

    for epoch in range(args.epochs):
        epoch_metrics: Dict[str, float] = {}
        start_time = time.time()

        for step in range(args.steps_per_epoch):
            # Sample batch from PushTMiniDataset or generate synthetic sequence
            if dataset is not None:
                batch = dataset.sample_batch(
                    args.batch_size, args.history_size, args.num_preds
                )
            else:
                batch = generate_synthetic_batch(
                    args.batch_size, seq_len, args.img_size, action_dim
                )

            # Run one train step: forward, backward, optimizer update, and metrics
            metrics = train_step(batch)

            # Evaluate arrays and state to trigger lazy execution and materialize updated parameters
            mx.eval(metrics, state)

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v.item()

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(
                    f"  Step {step + 1}/{args.steps_per_epoch} | "
                    f"Loss: {metrics['loss'].item():.4f} | "
                    f"Pred: {metrics['pred_loss'].item():.6f} | "
                    f"CosSim: {metrics['cos_sim'].item():+.3f} | "
                    f"GradNorm: {metrics['grad_norm'].item():.4f}"
                )

        elapsed = time.time() - start_time
        avg_metrics = {
            k: v / args.steps_per_epoch for k, v in epoch_metrics.items()
        }

        # Evaluate on fixed held-out probe if requested
        val_telemetry = {}
        if eval_batch is not None:
            _, val_dict = loss_fn(model, eval_batch)
            mx.eval(val_dict)
            val_telemetry = {
                f"val_{k}": v.item() for k, v in val_dict.items()
            }

        # Compose informative console log
        log_parts = [
            f"Epoch {epoch + 1}/{args.epochs}",
            f"Loss: {avg_metrics['loss']:.4f}",
            f"Pred: {avg_metrics['pred_loss']:.6f}",
            f"SIGReg: {avg_metrics['sigreg_loss']:.4f}",
            f"CosSim: {avg_metrics['cos_sim']:+.3f}",
            f"EmbStd: {avg_metrics['emb_std']:.4f}",
            f"GradNorm: {avg_metrics['grad_norm']:.4f}",
        ]
        if args.grad_breakdown:
            log_parts.append(
                f"(enc: {avg_metrics['enc_grad_norm']:.3f}, "
                f"pred: {avg_metrics['pred_grad_norm']:.3f}, "
                f"act: {avg_metrics['act_grad_norm']:.3f}, "
                f"proj: {avg_metrics['proj_grad_norm']:.3f})"
            )
        if val_telemetry:
            log_parts.append(
                f"ValLoss: {val_telemetry['val_loss']:.4f} | "
                f"ValPred: {val_telemetry['val_pred_loss']:.6f} | "
                f"ValCos: {val_telemetry['val_cos_sim']:+.3f}"
            )
        log_parts.append(f"Time: {elapsed:.2f}s")
        print(" | ".join(log_parts))

        # Output to structured JSON Lines log if path specified
        if args.metrics_path:
            record = {
                "epoch": epoch + 1,
                "elapsed": elapsed,
                **avg_metrics,
                **val_telemetry,
            }
            with open(args.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    # Save model weights
    print(f"Saving weights to {args.save_path}...")
    model.save_weights(args.save_path)
    print("Training verification complete!")


if __name__ == "__main__":
    main()
