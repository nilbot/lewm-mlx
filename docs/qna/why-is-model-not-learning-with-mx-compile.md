# Why Is the Model Not Learning When Training With mx.compile in MLX?

## Context
During training of the Le World Model (`lewm_mlx/train.py`) on Push-T demonstration trajectories, a 200-epoch training run was launched with `@mx.compile` decorating the training step function:
```python
loss_and_grads = nn.value_and_grad(model, loss_fn)

@mx.compile
def train_step(batch):
    return loss_and_grads(model, batch)
```
Across all 200 epochs (1,000 optimization steps), the loss remained frozen at exactly $0.961301 \to 0.961303$, fluctuating only at the 5th decimal place due to stochastic batch sampling. The code threw no errors, GPU execution was fast ($0.08\text{s}$ per epoch), and `optimizer.update(model, grads)` was executing every step—yet the model failed to learn anything.

## Answer

### 1. The Compiler Closure Capture Mechanism
Unlike PyTorch (which operates in eager mode or has a dynamic tracer in `torch.compile` that tracks module parameter pointers) and unlike JAX (which requires pure functions with explicit parameter passing `grads = grad(f)(params, x)`), MLX adopts a hybrid stateful module design with a graph compiler:

When a function is compiled with `@mx.compile` without specifying `inputs` and `outputs`:
1. MLX traces the computation graph using only the **explicit function arguments** (`batch`).
2. Any `mx.array` referenced inside the function from the enclosing Python scope (such as parameters in `model`) is treated as a **static compile-time constant**, baked directly into the computation graph at initial compilation time ($\theta_0$).
3. When `optimizer.update(model, grads)` subsequently updates parameter arrays in Python memory, the compiled Metal execution graph **does not re-read or bind to the updated weights**. It continues executing the static graph compiled at step 0.

### 2. Why It Fails Silently
This failure mode is particularly deceptive because:
- **No runtime exceptions are raised**: MLX treats the constant graph evaluation as perfectly valid.
- **Weights in memory actually change**: Inspecting `model.parameters()` in Python reveals that weights *are* changing because `optimizer.update` applies gradients computed by the step.
- **Loss and gradients remain frozen**: Because the forward and backward passes execute against $\theta_0$:
  $$\mathcal{L}_t = \mathcal{L}(\theta_0, \mathbf{x}_t)$$
  $$\mathbf{g}_t = \nabla_\theta \mathcal{L}(\theta_0, \mathbf{x}_t)$$
  The forward pass logs the loss of the random initial model on every step, and the optimizer repeatedly applies gradient steps evaluated at the initial point $\theta_0$ rather than along the optimization trajectory $\theta_t$.

### 3. The Canonical MLX Solution: Stateful Compilation
To compile a stateful training step in MLX, the mutable state (both `model.state` and `optimizer.state`) must be explicitly declared as both `inputs` and `outputs` to `mx.compile`:

```python
loss_and_grads = nn.value_and_grad(model, loss_fn)

def step_fn(batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, mx.array]]:
    (loss, metrics), grads = loss_and_grads(model, batch)
    optimizer.update(model, grads)
    return loss, metrics

# Register both model parameters and optimizer momentum/velocity buffers
state = [model.state, optimizer.state]
train_step = mx.compile(step_fn, inputs=state, outputs=state)
```

In the training loop, pass `state` to `mx.eval` alongside metrics:
```python
loss, metrics = train_step(batch)
# Force lazy evaluation of metrics, weights, and optimizer buffers
mx.eval(loss, metrics, state)
```

### 4. Rule of Thumb for MLX Compilation
- **Pure functions (e.g., mathematical transformations, loss reductions)**: Use `@mx.compile`.
- **Functions reading or writing `nn.Module` parameters or `Optimizer` states**: **Never** use bare `@mx.compile`. Always pass `inputs=state, outputs=state` with `state = [model.state, optimizer.state]`.
- **Instrumentation guard**: Always monitor global gradient norm $\|\mathbf{g}\|_2$ and target prediction cosine similarity $\cos(\hat{\mathbf{s}}, \mathbf{s})$. If $\cos(\hat{\mathbf{s}}, \mathbf{s})$ fails to rise above $\approx 0.0$ while loss remains invariant across epochs, verify that the compiled graph is bound to the live parameter state.

Related: a compiled graph can also be correctly bound while the representation still fails to learn. Projector `LayerNorm` heads collapse embeddings to a constant (`EmbStd` ~ 0.0002, `Pred` ~ 0); see [2026-09-17-train-collapse-layernorm-projectors.md](../journal/2026-09-17-train-collapse-layernorm-projectors.md).
