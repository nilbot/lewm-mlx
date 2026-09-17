"""Batch-size ladder for the 224px paper-scale LeWM model.

Measures per-step wall time for the full recipe at several batch sizes, each in
its own subprocess so a memory cliff at one size cannot disturb the others. Run
after a paper-scale training finishes (it needs the machine's full memory
budget). The reported `peak_rss_gb` comes from `resource.getrusage` and does not
capture MLX's Metal-backed allocations; the timing cliff is the observable.

Usage:
    uv run python scripts/batch_ladder.py
"""

import argparse
import json
import resource
import subprocess
import sys
import time

BATCHES = (32, 40, 48, 56, 64)


def run_single(batch_size: int, steps: int) -> None:
    """Benchmarks one batch size and prints a JSON line with time and peak RSS."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt

    from lewm_mlx.jepa import JEPA
    from lewm_mlx.module import ARPredictor, Embedder, MLP, SIGReg
    from lewm_mlx.vit import ViTModel

    encoder = ViTModel(
        image_size=224,
        patch_size=14,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    )
    predictor = ARPredictor(
        num_frames=3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=192,
        hidden_dim=192,
        dim_head=64,
        dropout=0.1,
    )
    action_encoder = Embedder(input_dim=10, smoothed_dim=192, emb_dim=192)
    projector = MLP(
        input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm
    )
    pred_proj = MLP(
        input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm
    )
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.train()
    sigreg = SIGReg(knots=17, num_proj=1024)
    optimizer = opt.AdamW(learning_rate=5e-5, weight_decay=1e-3)

    def loss_fn(model, batch):
        out = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
        emb, act_emb = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, :3], act_emb[:, :3])
        pred_loss = mx.mean(mx.square(pred - emb[:, 1:]))
        total = pred_loss + 0.09 * sigreg(emb.transpose(1, 0, 2))
        return total, {"pred": pred_loss}

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def step(batch):
        (loss, metrics), grads = value_and_grad(model, batch)
        optimizer.update(model, grads)
        return loss, metrics

    state = [model.state, optimizer.state]
    compiled = mx.compile(step, inputs=state, outputs=state)

    pixels = mx.random.uniform(shape=(batch_size, 4, 3, 224, 224))
    actions = mx.random.normal(shape=(batch_size, 4, 10))
    mx.eval(pixels, actions)
    batch = {"pixels": pixels, "action": actions}

    loss, metrics = compiled(batch)
    mx.eval(loss, metrics, state)  # trace and warm up

    started = time.time()
    for _ in range(steps):
        loss, metrics = compiled(batch)
        mx.eval(loss, metrics, state)
    per_step = (time.time() - started) / steps
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(
        json.dumps(
            {
                "batch": batch_size,
                "s_per_step": round(per_step, 3),
                "peak_rss_gb": round(peak_gb, 2),
                "loss": round(loss.item(), 4),
            }
        ),
        flush=True,
    )


def main() -> None:
    """Runs one size (with --batch) or the whole ladder in subprocesses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.batch:
        run_single(args.batch, args.steps)
        return
    for batch_size in BATCHES:
        print(f"--- batch {batch_size}", flush=True)
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--batch",
                str(batch_size),
                "--steps",
                str(args.steps),
            ]
        )


if __name__ == "__main__":
    main()
