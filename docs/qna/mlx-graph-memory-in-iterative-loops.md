# How to Manage MLX Computation Graph Accumulation in Iterative Optimization?

## Context
During the implementation of the Cross-Entropy Method (`CEMPlanner`) for goal-directed latent trajectory optimization, candidate actions $\mathbf{A} \sim \mathcal{N}(\mu, \sigma^2)$ are evaluated across $N$ iterations (typically $N \in [5, 10]$) with $S$ candidate trajectories ($S = 128$). In initial profiling, intermediate memory consumption grew monotonically across loop iterations, and execution speed degraded as the planning horizon increased.

## Answer

### 1. Lazy Evaluation and Graph Accumulation
Unlike PyTorch, which operates primarily in eager mode by default, MLX employs lazy evaluation. Operations recorded on `mx.array` instances do not execute immediately; rather, they append nodes to a computation graph that is evaluated only when data is explicitly read, formatted, or requested by an evaluation barrier.

In iterative optimization loops such as CEM:
$$\mu_{k+1} = \alpha \mu_k + (1 - \alpha) \frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} \mathbf{A}_{k, i}$$
$$\sigma_{k+1} = \alpha \sigma_k + (1 - \alpha) \sqrt{\frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} (\mathbf{A}_{k, i} - \bar{\mathbf{A}}_{k, \mathcal{E}})^2 + \epsilon}$$

If $\mu$ and $\sigma$ are updated across $N$ iterations without forcing evaluation, MLX constructs a nested computational graph linking all $N$ iterations together. This produces two adverse effects:
1. **Memory Bloat**: Unified memory retains all intermediate candidate tensors $\mathbf{A}_k$ of shape $[B, S, T, D]$ and intermediate model rollouts.
2. **Evaluation Latency**: When a downstream step finally queries `best_plan` or `best_cost`, MLX must evaluate the accumulated multi-iteration graph in a single monolithic dispatch, increasing scheduling overhead on the Metal GPU.

### 2. The Solution: Periodic Materialization Barrier
To prevent graph expansion, introduce explicit evaluation barriers using `mx.eval` at the iteration boundary:

```python
# Update distribution parameters with Polyak momentum alpha
if iter_idx < self.iterations - 1:
    batch_e_mean = mx.stack(elite_means, axis=0)  # [B, T, D]
    batch_e_std = mx.stack(elite_stds, axis=0)    # [B, T, D]
    mu = self.alpha * mu + (1.0 - self.alpha) * batch_e_mean
    sigma = self.alpha * sigma + (1.0 - self.alpha) * batch_e_std
    
    # Materialize arrays and release previous iteration graph nodes
    mx.eval(mu, sigma)
```

Calling `mx.eval(mu, sigma)` forces immediate scheduling and dispatch to the Apple Silicon GPU, materializing the numerical values for $\mu$ and $\sigma$ and allowing garbage collection to reclaim the candidate arrays $\mathbf{A}_k$ from unified memory.

### 3. Boundary Condition Optimization
On the final iteration ($k = N - 1$), the best trajectory $\mathbf{A}^*$ and cost $J^*$ have already been extracted from the candidate pool. The standard deviation $\sigma_{N}$ and updated mean $\mu_{N}$ are never sampled again. Guarding parameter updates with `if iter_idx < self.iterations - 1:` avoids computing redundant square roots and reductions on the final step:

```python
# Only compute elite variance and update parameters if subsequent iteration exists
if iter_idx < self.iterations - 1:
    e_mean = mx.mean(elite_candidates, axis=0)  # [T, D]
    diff_sq = mx.square(elite_candidates - e_mean)  # [K, T, D]
    e_std = mx.sqrt(mx.mean(diff_sq, axis=0) + 1e-6)  # [T, D]
    elite_means.append(e_mean)
    elite_stds.append(e_std)
```
