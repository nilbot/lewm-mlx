"""Training pipeline for Le World Model in MLX.

Trains a Joint-Embedding Predictive Architecture (JEPA) on visual demonstration
trajectories from Push-T (PushTMiniDataset) or procedurally generated synthetic
trajectories using Self-Information-Gauged Regularization (SIGReg) and latent prediction loss.
"""

import argparse
import time
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from lewm_mlx.dataset import PushTMiniDataset
from lewm_mlx.jepa import JEPA
from lewm_mlx.module import ARPredictor, Embedder, MLP, SIGReg
from lewm_mlx.vit import ViTModel


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
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        r"""Computes combined latent prediction and SIGReg regularization loss.

        Loss formulation:
            $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{SIGReg}} \mathcal{L}_{\text{SIGReg}}$$
            $$\mathcal{L}_{\text{pred}} = \frac{1}{B \cdot H \cdot D} \| \hat{\mathbf{s}}_{1:H} - \mathbf{s}_{K:H+K} \|_2^2$$

        Args:
            model: JEPA model instance.
            batch: Dictionary containing "pixels" and "action" MLX tensors:
                - pixels: $[B, T, 3, H, W]$
                - action: $[B, T, D_{\text{act}}]$

        Returns:
            Tuple of (total_loss, (pred_loss, sigreg_loss)):
                total_loss: Optimization scalar objective $\mathcal{L}_{\text{total}}$.
                pred_loss: Mean squared prediction error in latent space.
                sigreg_loss: Empirical self-information gauge regularization loss.
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

        pred_loss = mx.mean(mx.square(pred_emb - tgt_emb))
        # SIGReg expects [T, B, D] for marginal regularization over batch
        sigreg_loss = sigreg(emb.transpose(1, 0, 2))

        total_loss = pred_loss + args.sigreg_weight * sigreg_loss
        return total_loss, (pred_loss, sigreg_loss)

    # Step function compiled for MLX GPU/Metal execution
    loss_and_grads = nn.value_and_grad(model, loss_fn)

    @mx.compile
    def train_step(batch: Dict[str, mx.array]):
        return loss_and_grads(model, batch)

    print("Starting MLX Le World Model Training Loop...")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Steps per epoch: {args.steps_per_epoch}")
    print(f"Batch size: {args.batch_size}, Image size: {args.img_size}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_sigreg_loss = 0.0
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

            # Run one train step: forward, backward, gradients
            (loss, (pred_loss, sigreg_loss)), grads = train_step(batch)

            # Update parameters via AdamW
            optimizer.update(model, grads)

            # Evaluate arrays to trigger lazy execution and materialize updated parameters
            mx.eval(loss, pred_loss, sigreg_loss, model)

            epoch_loss += loss.item()
            epoch_pred_loss += pred_loss.item()
            epoch_sigreg_loss += sigreg_loss.item()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {epoch_loss / args.steps_per_epoch:.6f} | "
            f"Pred Loss: {epoch_pred_loss / args.steps_per_epoch:.6f} | "
            f"SIGReg Loss: {epoch_sigreg_loss / args.steps_per_epoch:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

    # Save model weights
    print(f"Saving weights to {args.save_path}...")
    model.save_weights(args.save_path)
    print("Training verification complete!")


if __name__ == "__main__":
    main()
