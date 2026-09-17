# 2026-09-17: Representation Collapse from LayerNorm Projection Heads

## Anomaly

After the `mx.compile` state-binding fix (see
[2026-09-17-train-compile-state-fix.md](2026-09-17-train-compile-state-fix.md)),
training no longer ran on frozen initial weights, but the latent representation
collapsed instead of learning. Two runs from the same day:

- Batch 16, 50 epochs x 20 steps (`--num-episodes 200 --lr 1e-4`):
  `Loss: 0.6640 -> 0.5789`, `Pred Loss: 0.058107 -> 0.000000`, `SIGReg Loss:
  6.7322 -> 6.4328`, `EmbStd: 0.0005 -> 0.0002`.
- Batch 128, 10 epochs x 20 steps (same flags):
  `Loss: 4.9058 -> 4.6316`, `Pred: 0.060014 -> 0.000000`, `SIGReg: 53.8421 ->
  51.4621`, `CosSim: -0.027 -> -0.015`, `EmbStd: 0.0004 -> 0.0002`.

The prediction loss reaching zero while cosine similarity stays near zero and
embedding dispersion vanishes is the signature of a trivial solution, not of
learning: the encoder maps every frame to a single constant vector, so the
predictor's target is trivially predictable.

## Investigation

1. **SIGReg floor identifies the collapse.** Evaluating `SIGReg` directly on a
   constant (zero) embedding tensor returns exactly `6.4328` for `B = 16` and
   `51.4621` for `B = 128`, with zero gradient norm. These are precisely the
   frozen values observed during training, so the regularizer was no longer
   acting on a live distribution.
2. **PyTorch ablation isolates the cause.** With the reference model code
   (`le-wm-ref`) and the same Push-T batches, 5 epochs at `B = 16`:

   | Projection heads | ImageNet norm | EmbNorm (ep 1 -> ep 5) | EmbStd | CosSim |
   |---|---|---|---|---|
   | LayerNorm | off (old MLX config) | 0.7510 -> 0.0284 | 0.0008 -> 0.0006 | 0.6027 -> 0.7878 |
   | LayerNorm | on | 0.7442 -> 0.0298 | 0.0016 -> 0.0012 | 0.6060 -> 0.7448 |
   | BatchNorm | off | 2.7570 -> 4.4820 | 0.3502 -> 0.5799 | 0.1560 -> 0.7950 |
   | BatchNorm | on (reference) | 2.9306 -> 4.4788 | 0.3695 -> 0.5824 | 0.1675 -> 0.7900 |

   LayerNorm projection heads collapse regardless of frame normalization;
   BatchNorm heads keep embedding norm and dispersion growing. The decisive
   deviation was therefore the MLX-only choice of `nn.LayerNorm` in the
   `projector` / `pred_proj` MLPs. The PyTorch reference uses `BatchNorm1d`
   (see `le-wm-ref/config/train/model/lewm.yaml` and `le-wm-ref/module.py`).
3. **Frame standardization was also missing.** The reference standardizes
   frames with ImageNet channel statistics inside its transform pipeline
   (`le-wm-ref/utils.py::get_img_preprocessor`); the MLX pipeline fed raw
   `[0, 1]` frames to the encoder. The ablation above shows BatchNorm alone
   already prevents collapse at this scale, but restoring the reference
   normalization keeps train/inference preprocessing identical.

## Fix

- `lewm_mlx/train.py` now defaults to BatchNorm projection heads via
  `--norm-fn {batchnorm,layernorm,none}` and applies ImageNet frame
  standardization by default (`--no-imagenet-norm` restores the old behavior).
- The fixed evaluation probe now runs under `model.eval()` and restores
  `model.train()`, so BatchNorm uses running statistics for validation.
- `demo.py` mirrors the same two flags, and both entry points share
  `lewm_mlx/preprocessing.py::imagenet_normalize` so the encoder sees identical
  input distributions at training and inference time.

## Verification

### 2x2 ablation (real `train.py`, failing command, 10 epochs x 20 steps, B = 128)

| Projection heads | ImageNet norm | Epoch 1 (Loss / Pred / CosSim / EmbStd) | Epoch 10 (Loss / Pred / CosSim / EmbStd) |
|---|---|---|---|
| LayerNorm | off | 4.8653 / 0.068927 / -0.075 / 0.0004 | 4.6316 / 0.000000 / -0.002 / 0.0002 |
| LayerNorm | on | 4.9972 / 0.076287 / -0.016 / 0.0008 | 4.6316 / 0.000000 / +0.028 / 0.0002 |
| BatchNorm | off | 2.2553 / 0.338997 / +0.246 / 0.5269 | 0.1877 / 0.094902 / +0.923 / 0.7766 |
| BatchNorm | on (default) | 1.8393 / 0.357748 / +0.250 / 0.5693 | 0.2146 / 0.115949 / +0.899 / 0.7672 |

LayerNorm collapses to the same SIGReg floor regardless of frame normalization,
so the fix is not an artifact of changing the input distribution; the causal
factor is the projection-head normalization. BatchNorm learns with or without
ImageNet standardization, which is retained for reference parity.

### Objective integrity

- The loss function diff against the pre-fix commit touches only the pixel
  standardization block; `pred_loss`, `sigreg_loss = sigreg(emb.transpose(1, 0,
  2))`, and `total_loss = pred_loss + args.sigreg_weight * sigreg_loss` are
  unchanged, and match `le-wm-ref/train.py::lejepa_forward` term for term
  (same slices, MSE, λ = 0.09 from `config/train/lewm.yaml`).
- Across all nine logged runs (10 and 50 epochs, both heads, both batch sizes),
  `|loss - (pred_loss + 0.09 * sigreg_loss)| <= 2.7e-7`, i.e. float32 noise.
- A two-epoch run with `--sigreg-weight 0.5` satisfies
  `|loss - (pred_loss + 0.5 * sigreg_loss)| <= 1.9e-8` and rejects the 0.09
  coefficient by 7.06, showing λ is applied exactly as configured rather than
  hard-coded to fit the fix.
- MLX and PyTorch `SIGReg` agree on identical unit-variance embeddings
  (`[T=4, B=128, D=64]`): MLX 1.0506 ± 0.0317 vs PyTorch 1.0695 ± 0.0258 at
  `num_proj=256`; 1.0475 ± 0.0114 vs 1.0511 ± 0.0164 at 1024.

### Effectiveness beyond loss values

- Batch 128, 50 epochs (final tree): `Loss 0.1251 | Pred 0.057037 | SIGReg
  0.7564 | CosSim +0.952 | ValCos +0.962 | EmbStd 0.8120`.
- Batch 16, 50 epochs (matching the first failing log above): `Loss 0.0797 |
  Pred 0.026666 | SIGReg 0.5897 | CosSim +0.973 | ValCos +0.987 | EmbStd
  0.6725`.
- Held-out batch (64 sequences, trained 50-epoch weights): pred MSE 0.0551,
  cosine +0.952, target norm 6.429, `R^2 = 0.922` against a zero predictor.
  Randomly initialized weights score MSE 0.0782 with cosine -0.064; the
  collapse configuration scores MSE 0.000000 with `EmbStd` 0.0002.
- Five-step autoregressive rollout beats a persistence baseline (repeat the
  last context embedding) at horizons 2-5: mean MSE 0.1035 vs 0.3783, cosine
  +0.98 vs +0.83 at the final step. At horizon 1 persistence is naturally
  strong (0.0691 vs 0.1299), which reflects the small frame-to-frame motion of
  Push-T and the 1-step training objective, not a degenerate solution.

`uv run pytest tests/` passes with 33 tests (29 existing plus new preprocessing
and checkpoint-loading coverage). `demo.py --weights <batch-norm checkpoint>`
loads and runs end-to-end; on a fixed trajectory, mean rollout MSE is 0.0663
with the trained (ImageNet-normalized) inputs versus 0.5329 for raw `[0, 1]`
frames, confirming the inference-side preprocessing is required. CEM planning
reduces its objective from 15.040411 (random shooting) to 2.157720 (85.65%).

## Follow-Up

- Checkpoints trained with the pre-fix LayerNorm heads must be evaluated with
  `--norm-fn layernorm --no-imagenet-norm`; the new defaults will otherwise
  reject them with a parameter mismatch.
- `lewm_mlx/train.py` still builds `SIGReg(knots=17, num_proj=256)` while the
  reference config uses `num_proj=1024`. The statistic's expectation is
  identical, but the Monte-Carlo estimator variance is roughly 2-3x larger
  (0.03 vs 0.01-0.02 on the probe above); aligning it is a separate fidelity
  decision.
- The earlier compile-fix journal's headline result (`Pred Loss 0.000538` at
  epoch 10) was itself the collapse signature reported here; prediction loss
  alone is not evidence of representation learning. Use `CosSim`, `EmbStd`,
  and `SIGReg` jointly as health checks.
