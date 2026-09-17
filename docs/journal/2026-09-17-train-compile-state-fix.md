# 2026-09-17: Resolution of Invariant Training Loss and `mx.compile` Closure State

## Anomaly
During training execution via `lewm_mlx/train.py`:
```bash
uv run python lewm_mlx/train.py \
    --dataset pusht_mini \
    --num-episodes 5000 \
    --epochs 200 \
    --steps-per-epoch 5 \
    --img-size 96 \
    --save-path lewm_weights.npz
```
the training loss remained virtually constant across all 200 epochs:
- Epoch 1: `Loss: 0.961301 | Pred Loss: 0.232007 | SIGReg Loss: 8.103267`
- Epoch 200: `Loss: 0.961303 | Pred Loss: 0.232014 | SIGReg Loss: 8.103214`

## Root Cause Investigation
1. **Isolated Step Analysis**: Evaluating a single fixed batch without compilation revealed that gradient descent and optimizer updates functioned normally (MSE loss dropping from 0.29 to 0.001 within 40 steps).
2. **`mx.compile` Closure Capture**:
   In `train.py`, the training step was compiled as:
   ```python
   loss_and_grads = nn.value_and_grad(model, loss_fn)

   @mx.compile
   def train_step(batch):
       return loss_and_grads(model, batch)
   ```
   In MLX, `@mx.compile` creates an optimized computation graph based solely on function arguments (`batch`). When `model` is accessed as a closure variable from the enclosing scope without registering `model.state` in `inputs` and `outputs`, MLX treats the parameter arrays of `model` as static constants at compile time.
3. **Symptom Mechanism**: Even though `optimizer.update(model, grads)` updated model parameters in Python memory, `train_step` re-executed the static graph compiled with the initial random weights $\theta_0$. Consequently:
   - Forward predictions and losses were evaluated on the initial untrained weights $\theta_0$ on every step.
   - Gradients were always computed at $\nabla_\theta \mathcal{L}(\theta_0)$, preventing meaningful parameter optimization along the trajectory $\theta_t$.

## Solution
Adopted the canonical MLX stateful compilation pattern by bundling model and optimizer state in `inputs` and `outputs`:
```python
loss_and_grads = nn.value_and_grad(model, loss_fn)

def step_fn(batch: Dict[str, mx.array]) -> Tuple[mx.array, mx.array, mx.array]:
    (loss, (pred_loss, sigreg_loss)), grads = loss_and_grads(model, batch)
    optimizer.update(model, grads)
    return loss, pred_loss, sigreg_loss

state = [model.state, optimizer.state]
train_step = mx.compile(step_fn, inputs=state, outputs=state)
```
In the training loop:
```python
loss, pred_loss, sigreg_loss = train_step(batch)
mx.eval(loss, pred_loss, sigreg_loss, state)
```

## Outcome
Running the Push-T training loop with the fix demonstrated immediate and rapid convergence:
- Epoch 1: `Loss: 0.872273 | Pred Loss: 0.196954 | SIGReg Loss: 7.503544`
- Epoch 5: `Loss: 0.591077 | Pred Loss: 0.006970 | SIGReg Loss: 6.490081`
- Epoch 10: `Loss: 0.579757 | Pred Loss: 0.000538 | SIGReg Loss: 6.435773`
Latent prediction error dropped by over $360\times$ within 10 epochs (50 steps), while maintaining full Metal acceleration (~0.06s per epoch).
