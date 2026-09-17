# Paper-Scale Push-T Training Regime

## Goal

Train a LeWM model whose recipe matches the reference implementation
(`le-wm-ref`, LeWorldModel) as closely as this machine allows: an ~18M-parameter
model (ViT-tiny encoder + 6-layer autoregressive predictor), 224px frames with
ImageNet normalization, the two-term objective (next-embedding MSE plus
`0.09 * SIGReg`), and a 100-epoch budget over the full Push-T expert dataset.

## Hardware envelope

Apple M1 (8 cores), 16 GB unified memory, MLX 0.31.2. Measured throughput at
224px with the full recipe: **batch 32 = 1.7-1.8 s/step**; batch 64 falls off a
memory cliff (48 s/step), so batch 32 is the practical ceiling here.

## Recipe

| Setting | Reference (`le-wm-ref`) | This run | Note |
|---|---|---|---|
| Encoder | ViT-tiny 192/12L/3H/768, patch 14, 224px, from scratch | same | matches the reference config |
| Predictor | 6 layers, 16 heads, dim_head 64, mlp 2048, dropout 0.1 | same | |
| Projection heads | MLP 192 -> 2048 -> 192 with BatchNorm | same | |
| Objective | MSE + 0.09 x SIGReg (knots 17, num_proj 1024) | same | |
| Optimizer | AdamW, lr 5e-5, wd 1e-3, warmup + cosine, grad clip 1.0 | same | |
| Batch size | 128 | 32 | hardware ceiling |
| Steps per epoch | ~35 (one pass over 4433 windows) | 124 (one pass over 3957 training windows) | sample budget matched per epoch |
| Epochs | 100 | 100 | ~397k samples vs ~443k |
| Data | pusht_expert_train, 206 episodes, 90/10 split | same (185 train / 21 held out) | |
| Precision | bf16 | fp32 | M1 |
| Wall clock | "a few hours" on one GPU | approximately 6.3 h measured | 1.84 s/step |

## Launch command

```bash
uv run python lewm_mlx/train.py \
    --dataset pusht_mini --num-episodes 206 --frameskip 5 \
    --img-size 224 --patch-size 14 --embed-dim 192 \
    --encoder-layers 12 --encoder-heads 3 --encoder-mlp-dim 768 \
    --predictor-depth 6 --predictor-heads 16 --predictor-mlp-dim 2048 \
    --predictor-dim-head 64 --predictor-dropout 0.1 \
    --proj-hidden 2048 --sigreg-num-proj 1024 --sigreg-weight 0.09 \
    --history-size 3 --num-preds 1 \
    --batch-size 32 --steps-per-epoch 124 --epochs 100 \
    --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0 \
    --lr-schedule cosine --warmup-epochs 5 --seed 3072 \
    --val-fraction 0.1 --eval-fixed \
    --metrics-path outputs/paper-pusht-20260917/metrics.jsonl \
    --save-path outputs/paper-pusht-20260917/lewm.npz --save-every 10
```

Artifacts land under `outputs/paper-pusht-20260917/`: `train.log`,
`metrics.jsonl`, the final weights, and a checkpoint every 10 epochs.

## Evaluation plan

1. Epoch curves from `metrics.jsonl`: `val_cos_sim`, `val_pred_loss`, `sigreg_loss`,
   and `emb_std` as a collapse screen.
2. Five-step latent rollout MSE against a persistence baseline at 224px
   (`demo.py --mode rollout --img-size 224`).
3. CEM planning cost reduction against the random-shooting baseline on held-out
   episodes.
4. Follow-up for the paper's headline metric: planning success rate in the
   PushT environment requires integrating the `stable_worldmodel` environment,
   which this repository does not yet include.

## Caveats

- Batch 32 differs from the reference batch 128: the sample budget is matched,
  but the per-step gradient noise and the batch statistics seen by BatchNorm and
  SIGReg differ.
- fp32 instead of bf16, and M1 throughput instead of a datacenter GPU, stretch
  the wall clock from "a few hours" to approximately 6.3 hours.
