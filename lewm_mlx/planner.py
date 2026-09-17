"""Vectorised Cross-Entropy Method (CEM) and Random Shooting Planners in MLX.

Implements model-predictive control (MPC) trajectory optimizers that operate
directly within the latent representation space of the Joint-Embedding Predictive
Architecture (JEPA). The optimizer evaluates candidate action sequences in parallel
and iteratively fits a diagonal Gaussian belief distribution over optimal actions.
"""

from typing import Any, Dict, List, Tuple
import mlx.core as mx
from lewm_mlx.jepa import JEPA


class CEMPlanner:
    r"""Cross-Entropy Method (CEM) trajectory optimizer in MLX.

    Optimizes open-loop action sequences by iteratively sampling candidates from a
    diagonal Gaussian distribution, evaluating candidate rollout costs through the JEPA
    latent predictive model, and refitting the distribution to the top elite candidates
    with Polyak momentum smoothing:

    .. math::
        \min_{\mathbf{A}} \mathcal{J}(\mathbf{A}) = \left\| \hat{\mathbf{z}}_T(\mathbf{A}) - \mathbf{z}_{\text{goal}} \right\|_2^2

    Candidate action trajectories are sampled as:

    .. math::
        \mathbf{A}^{(k)} \sim \mathcal{N}(\mu, \operatorname{diag}(\sigma^2)), \quad k = 1, \dots, S

    and clipped to control bounds $[\mathbf{a}_{\min}, \mathbf{a}_{\max}]$.

    The distribution parameters are updated with momentum factor $\alpha \in [0, 1]$:

    .. math::
        \mu \leftarrow \alpha \mu + (1 - \alpha) \frac{1}{K} \sum_{k \in \mathcal{E}} \mathbf{A}^{(k)}
    .. math::
        \sigma \leftarrow \alpha \sigma + (1 - \alpha) \sqrt{\frac{1}{K} \sum_{k \in \mathcal{E}} \left( \mathbf{A}^{(k)} - \mu_{\text{elite}} \right)^2 + \epsilon}
    """

    def __init__(
        self,
        model: JEPA,
        planning_horizon: int = 5,
        action_dim: int = 10,
        num_samples: int = 128,
        num_elites: int = 16,
        iterations: int = 5,
        alpha: float = 0.1,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
    ) -> None:
        r"""Initializes the CEM planner with optimization hyperparameters.

        Args:
            model: JEPA model instance providing `get_cost(info_dict, action_candidates)`.
            planning_horizon: Temporal length of planned action sequence ($T$).
            action_dim: Dimensionality of individual control action vectors ($D$).
            num_samples: Number of action candidates evaluated per iteration ($S$).
            num_elites: Number of top-performing candidates used to refit distribution ($K$).
            iterations: Number of optimization refinement iterations ($N_{\text{iter}}$).
            alpha: Momentum coefficient for exponential moving average updates ($\alpha$).
            lower_bound: Lower saturation bound for action coordinates ($a_{\min}$).
            upper_bound: Upper saturation bound for action coordinates ($a_{\max}$).

        Raises:
            ValueError: If any dimensionality or hyperparameter is outside valid domains.
        """
        if planning_horizon <= 0:
            raise ValueError(
                f"planning_horizon must be positive, got {planning_horizon}"
            )
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_elites <= 0:
            raise ValueError(f"num_elites must be positive, got {num_elites}")
        if num_elites > num_samples:
            raise ValueError(
                f"num_elites ({num_elites}) cannot exceed num_samples ({num_samples})"
            )
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if lower_bound > upper_bound:
            raise ValueError(
                f"lower_bound ({lower_bound}) cannot be greater than upper_bound ({upper_bound})"
            )

        self.model = model
        self.planning_horizon = planning_horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.alpha = alpha
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def plan(
        self, info_dict: Dict[str, Any]
    ) -> Tuple[mx.array, float, List[float]]:
        r"""Executes vectorised CEM trajectory optimization.

        Args:
            info_dict: Observation dictionary containing:
                "pixels": Context observation frames $[B, S_{\text{ctx}}, H, C, H_{\text{img}}, W_{\text{img}}]$.
                "goal": Target observation frame $[B, S_{\text{goal}}, 1, C, H_{\text{img}}, W_{\text{img}}]$.

        Returns:
            best_plan: Optimal action sequence of shape $[T, D]$.
            best_cost: Minimal latent objective cost scalar.
            cost_history: History of lowest costs achieved across iterations.

        Raises:
            KeyError: If "pixels" or "goal" is not present in `info_dict`.
        """
        if "pixels" not in info_dict:
            raise KeyError("info_dict must contain 'pixels' key for context observations")
        if "goal" not in info_dict:
            raise KeyError("info_dict must contain 'goal' key for target observation")

        t_horizon = self.planning_horizon
        act_dim = self.action_dim
        s_samples = self.num_samples

        # Initialize distribution parameters: mu [T, D] (zeros), sigma [T, D] (0.5)
        mu = mx.zeros((t_horizon, act_dim))
        sigma = mx.full((t_horizon, act_dim), 0.5)

        cost_history: List[float] = []
        best_cost = float("inf")
        best_plan = mx.clip(mu, self.lower_bound, self.upper_bound)

        for _ in range(self.iterations):
            # Sample standard Gaussian noise: eps [1, S, T, D]
            eps = mx.random.normal(shape=(1, s_samples, t_horizon, act_dim))
            # Shift and scale: candidates [1, S, T, D]
            candidates = mu + sigma * eps
            candidates = mx.clip(candidates, self.lower_bound, self.upper_bound)

            # Evaluate candidate costs through JEPA forward rollout: costs [1, S]
            costs = self.model.get_cost(dict(info_dict), candidates)
            costs_flat = costs.squeeze(0)  # [S]

            # Identify elite indices with lowest cost
            elite_indices = mx.argsort(costs_flat)[: self.num_elites]
            elite_candidates = candidates[0, elite_indices]  # [K, T, D]

            iter_best_cost = costs_flat[elite_indices[0]].item()
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_plan = elite_candidates[0]  # [T, D]

            cost_history.append(best_cost)

            # Update distribution parameters with momentum alpha
            elite_mean = mx.mean(elite_candidates, axis=0)  # [T, D]
            diff_sq = mx.square(elite_candidates - elite_mean)  # [K, T, D]
            elite_std = mx.sqrt(mx.mean(diff_sq, axis=0) + 1e-6)  # [T, D]

            mu = self.alpha * mu + (1.0 - self.alpha) * elite_mean
            sigma = self.alpha * sigma + (1.0 - self.alpha) * elite_std

        return best_plan, best_cost, cost_history


class ShootingPlanner(CEMPlanner):
    r"""Random Shooting trajectory optimizer in MLX.

    Specialization of the Cross-Entropy Method with a single iteration and
    single elite ($N_{\text{iter}} = 1, K = 1$), evaluating $S$ randomly sampled
    candidate action sequences in a single forward pass and selecting the candidate
    with minimal cost.
    """

    def __init__(
        self,
        model: JEPA,
        planning_horizon: int = 5,
        action_dim: int = 10,
        num_samples: int = 128,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
    ) -> None:
        r"""Initializes the random shooting planner.

        Args:
            model: JEPA model instance providing `get_cost(info_dict, action_candidates)`.
            planning_horizon: Temporal length of planned action sequence ($T$).
            action_dim: Dimensionality of individual control action vectors ($D$).
            num_samples: Number of candidate action sequences sampled ($S$).
            lower_bound: Lower saturation bound for action coordinates ($a_{\min}$).
            upper_bound: Upper saturation bound for action coordinates ($a_{\max}$).
        """
        super().__init__(
            model=model,
            planning_horizon=planning_horizon,
            action_dim=action_dim,
            num_samples=num_samples,
            num_elites=1,
            iterations=1,
            alpha=0.0,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
