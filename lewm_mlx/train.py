import os
import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from lewm_mlx.vit import ViTModel
from lewm_mlx.module import ARPredictor, Embedder, MLP, SIGReg
from lewm_mlx.jepa import JEPA

def generate_synthetic_batch(batch_size, seq_len, img_size, action_dim):
    """Generates a synthetic batch of data for verification."""
    pixels = mx.random.normal(shape=(batch_size, seq_len, 3, img_size, img_size))
    # Add random NaNs to actions to simulate boundaries
    actions = mx.random.normal(shape=(batch_size, seq_len, action_dim))
    nan_mask = mx.random.uniform(shape=(batch_size, seq_len, 1)) < 0.05
    actions = mx.where(nan_mask, mx.array(float('nan')), actions)
    
    return {
        "pixels": pixels,
        "action": actions
    }

def main():
    parser = argparse.ArgumentParser(description="Train Le World Model in MLX")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--history-size", type=int, default=3, help="History context size")
    parser.add_argument("--num-preds", type=int, default=1, help="Number of steps to predict")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=20, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--sigreg-weight", type=float, default=0.09, help="SIGReg loss weight")
    parser.add_argument("--save-path", type=str, default="lewm_weights.npz", help="Path to save weights")
    args = parser.parse_args()

    # Define model sub-components
    # Tiny configuration for fast execution
    encoder = ViTModel(
        image_size=args.img_size,
        patch_size=16,
        num_channels=3,
        hidden_size=args.embed_dim,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256
    )
    
    predictor = ARPredictor(
        num_frames=args.history_size,
        depth=2,
        heads=2,
        mlp_dim=256,
        input_dim=args.embed_dim,
        hidden_dim=args.embed_dim
    )
    
    action_encoder = Embedder(
        input_dim=args.history_size * 2,  # Let's say action dim is 2, frameskip*action_dim
        smoothed_dim=args.embed_dim,
        emb_dim=args.embed_dim
    )
    
    projector = MLP(
        input_dim=args.embed_dim,
        hidden_dim=256,
        output_dim=args.embed_dim,
        norm_fn=nn.LayerNorm
    )
    
    pred_proj = MLP(
        input_dim=args.embed_dim,
        hidden_dim=256,
        output_dim=args.embed_dim,
        norm_fn=nn.LayerNorm
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj
    )
    model.train()

    sigreg = SIGReg(knots=17, num_proj=256)

    # Optimizer
    optimizer = opt.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Sequence length in dataset needs to be history_size + num_preds
    seq_len = args.history_size + args.num_preds
    action_dim = args.history_size * 2

    # Define loss function
    def loss_fn(model, batch):
        # Replace NaNs in action with 0.0
        actions = batch["action"]
        actions = mx.where(mx.isnan(actions), mx.array(0.0), actions)
        
        batch_new = {
            "pixels": batch["pixels"],
            "action": actions
        }

        output = model.encode(batch_new)
        emb = output["emb"]  # (B, T, D)
        act_emb = output["act_emb"]

        ctx_emb = emb[:, :args.history_size, :]
        ctx_act = act_emb[:, :args.history_size, :]

        tgt_emb = emb[:, args.num_preds:, :]
        pred_emb = model.predict(ctx_emb, ctx_act)

        pred_loss = mx.mean(mx.square(pred_emb - tgt_emb))
        sigreg_loss = sigreg(emb.transpose(1, 0, 2))
        
        total_loss = pred_loss + args.sigreg_weight * sigreg_loss
        return total_loss, (pred_loss, sigreg_loss)

    # Step function compiled for speed
    loss_and_grads = nn.value_and_grad(model, loss_fn)

    @mx.compile
    def train_step(batch):
        return loss_and_grads(model, batch)

    print("Starting MLX Le World Model Training Loop...")
    print(f"Epochs: {args.epochs}, Steps per epoch: {args.steps-per-epoch if hasattr(args, 'steps-per-epoch') else args.steps_per_epoch}")
    print(f"Batch size: {args.batch_size}, Image size: {args.img_size}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_sigreg_loss = 0.0
        start_time = time.time()

        for step in range(args.steps_per_epoch):
            # Generate synthetic sequence batch
            batch = generate_synthetic_batch(
                args.batch_size, seq_len, args.img_size, action_dim
            )
            
            # Run one train step
            (loss, (pred_loss, sigreg_loss)), grads = train_step(batch)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Evaluate arrays to trigger execution and materialize updated parameters
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
