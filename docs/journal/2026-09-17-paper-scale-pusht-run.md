# 2026-09-17: Paper-Scale Push-T Run and the Real Batch-Size Cliff

## What ran

The full 100-epoch paper recipe from
[../plans/2026-09-17-paper-scale-pusht-training.md](../plans/2026-09-17-paper-scale-pusht-training.md)
finished end to end: ViT-tiny 192/12L/3H/768 at 224px (patch 14) plus the
6-layer predictor, BatchNorm projection heads, ImageNet-normalized frames,
MSE + 0.09 x SIGReg, AdamW (lr 5e-5, wd 1e-3, warmup + cosine, grad clip 1.0),
batch 32, 124 steps/epoch, 185 train / 21 held-out episodes, seed 3072.
Artifacts: `outputs/paper-pusht-20260917/` with `train.log`, `metrics.jsonl`,
`lewm.npz`, and a checkpoint every 10 epochs.

| epoch | Loss | Pred | SIGReg | CosSim | EmbStd | ValPred | ValCos |
|---|---|---|---|---|---|---|---|
| 1 | 0.6260 | 0.1711 | 5.0542 | +0.486 | 0.4820 | 5.4947 | +0.790 |
| 100 | 0.0473 | 0.0049 | 0.4715 | +0.996 | 0.8550 | 0.0242 | +0.976 |

No collapse signature: `EmbStd` grew from 0.48 to 0.86 and stayed there while
`SIGReg` fell from 5.05 to 0.47 - the opposite of the LayerNorm collapse fixed
earlier the same day
([2026-09-17-train-collapse-layernorm-projectors.md](2026-09-17-train-collapse-layernorm-projectors.md)).

Epoch time: median 232 s, mean 280 s, min 206 s, max 862 s; 7.79 h summed
(~7.9 h wall clock). The 13 epochs above 350 s (worst: 46 = 862 s, 54 = 791 s,
53 = 716 s) track machine load, not the model - idle-machine epochs run at
206-233 s for the same 124 steps.

## The batch cliff is 32 -> 34, not 32 -> 64

The pre-run envelope measurement went straight from batch 32 (1.70 s/step) to
batch 64 (48 s/step), which read as "64 is the cliff". The post-run ladder
([scripts/batch_ladder.py](../../scripts/batch_ladder.py), random tensors, no
dataset, 2 timed steps after warm-up, one subprocess per size) puts the boundary
much lower:

| Batch | 32 | 34 | 36 | 38 | 40 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|---|
| s/step | 1.75 | 34.2 | 57.2 | 30.5 | 40.7 | 38.0 | 47.4 | 75.0 |

Batch 32 is the largest fast size; 34 already falls ~20x and everything above
sits in a 30-75 s/step paging regime with noise dominating (36 is slower than
48). `resource.getrusage` RSS is useless as a metric here - it reports
0.08-0.10 GB at every size and does not capture MLX's Metal-backed allocations;
the timing cliff is the observable.

Two conclusions: the trainer's batch-32 choice is the machine's actual ceiling,
not a conservative pick, and the earlier uint8 dataset redesign (0.76 GB) was
not what bounded batch size - the model working set is.

## Next

- Post-training evaluation is still open; the steps are listed in
  [../plans/2026-09-17-paper-scale-pusht-training.md](../plans/2026-09-17-paper-scale-pusht-training.md)
  under "Evaluation plan": five-step rollout MSE against a persistence baseline
  at 224px, then CEM planning cost reduction on held-out episodes.
- The paper's headline number (planning success rate) needs the
  `stable_worldmodel` PushT environment, which this repository does not include
  yet; `stable-worldmodel` and `stable-pretraining` are already declared in the
  dev extras.
