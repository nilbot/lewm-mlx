"""Unit tests for CEMPlanner and ShootingPlanner in MLX.

Tests trajectory optimization, cost monotonicity, action bounds clipping,
and input validation for vectorised cross-entropy method and random shooting planners.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from lewm_mlx.jepa import JEPA
from lewm_mlx.module import ARPredictor, Embedder, MLP
from lewm_mlx.planner import CEMPlanner, ShootingPlanner
from lewm_mlx.vit import ViTModel


def _build_dummy_jepa(
    img_size: int = 96,
    embed_dim: int = 32,
    history_size: int = 3,
    action_dim: int = 10,
) -> JEPA:
    """Builds a lightweight JEPA model for unit testing planners.

    Args:
        img_size: Square image resolution (H = W).
        embed_dim: Dimension of latent representations.
        history_size: Number of context frames in history.
        action_dim: Dimension of action vectors.

    Returns:
        Configured JEPA model in evaluation mode.
    """
    encoder = ViTModel(
        image_size=img_size,
        patch_size=16,
        num_channels=3,
        hidden_size=embed_dim,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    predictor = ARPredictor(
        num_frames=history_size,
        depth=1,
        heads=2,
        mlp_dim=64,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
    )
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=embed_dim,
        emb_dim=embed_dim,
    )
    projector = MLP(
        input_dim=embed_dim,
        hidden_dim=64,
        output_dim=embed_dim,
        norm_fn=nn.LayerNorm,
    )
    pred_proj = MLP(
        input_dim=embed_dim,
        hidden_dim=64,
        output_dim=embed_dim,
        norm_fn=nn.LayerNorm,
    )
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.eval()
    return model


def test_build_dummy_jepa():
    """Verifies helper function instantiates a functional JEPA model."""
    model = _build_dummy_jepa()
    assert isinstance(model, JEPA)


def test_cem_planner_execution():
    """Verifies CEMPlanner optimizes trajectories with correct shapes and monotonic cost tracking."""
    img_size = 96
    history_size = 3
    action_dim = 10
    planning_horizon = 5
    num_samples = 16
    num_elites = 4
    iterations = 4

    model = _build_dummy_jepa(
        img_size=img_size,
        history_size=history_size,
        action_dim=action_dim,
    )
    planner = CEMPlanner(
        model=model,
        planning_horizon=planning_horizon,
        action_dim=action_dim,
        num_samples=num_samples,
        num_elites=num_elites,
        iterations=iterations,
        alpha=0.1,
        lower_bound=-1.0,
        upper_bound=1.0,
    )

    # Context pixels: [1, 1, H, 3, img_size, img_size]
    # Goal pixels: [1, 1, 1, 3, img_size, img_size]
    init_pixels = mx.random.normal((1, 1, history_size, 3, img_size, img_size))
    goal_pixels = mx.random.normal((1, 1, 1, 3, img_size, img_size))

    info_dict = {"pixels": init_pixels, "goal": goal_pixels}
    best_actions, best_cost, cost_history = planner.plan(info_dict)

    # Assert best plan shape: (planning_horizon, action_dim)
    assert best_actions.shape == (planning_horizon, action_dim)

    # Assert best_cost is float
    assert isinstance(best_cost, float)
    assert not mx.isnan(mx.array(best_cost)).item()

    # Assert cost_history is a list of floats matching iterations
    assert isinstance(cost_history, list)
    assert len(cost_history) == iterations
    assert all(isinstance(c, float) for c in cost_history)

    # Assert cost monotonicity: best tracked cost is non-increasing across iterations
    for i in range(1, len(cost_history)):
        assert cost_history[i] <= cost_history[i - 1] + 1e-6, (
            f"Cost increased at iteration {i}: {cost_history[i]} > {cost_history[i - 1]}"
        )

    # Assert final best cost matches last recorded best cost
    assert abs(best_cost - cost_history[-1]) < 1e-6

    # Assert all values within [-1.0, 1.0]
    assert mx.all(best_actions >= -1.0).item()
    assert mx.all(best_actions <= 1.0).item()


def test_shooting_planner_execution():
    """Verifies ShootingPlanner operates as a single-iteration optimization."""
    img_size = 96
    history_size = 3
    action_dim = 10
    planning_horizon = 4
    num_samples = 16

    model = _build_dummy_jepa(
        img_size=img_size,
        history_size=history_size,
        action_dim=action_dim,
    )
    planner = ShootingPlanner(
        model=model,
        planning_horizon=planning_horizon,
        action_dim=action_dim,
        num_samples=num_samples,
        lower_bound=-1.0,
        upper_bound=1.0,
    )

    assert planner.iterations == 1
    assert planner.num_elites == 1

    init_pixels = mx.random.normal((1, 1, history_size, 3, img_size, img_size))
    goal_pixels = mx.random.normal((1, 1, 1, 3, img_size, img_size))

    info_dict = {"pixels": init_pixels, "goal": goal_pixels}
    best_actions, best_cost, cost_history = planner.plan(info_dict)

    assert best_actions.shape == (planning_horizon, action_dim)
    assert isinstance(best_cost, float)
    assert len(cost_history) == 1
    assert cost_history[0] == best_cost
    assert mx.all(best_actions >= -1.0).item()
    assert mx.all(best_actions <= 1.0).item()


def test_action_bounds_clipping():
    """Verifies actions are strictly clipped within arbitrary asymmetric bounds."""
    img_size = 96
    history_size = 3
    action_dim = 8
    planning_horizon = 4
    lower_bound = -0.35
    upper_bound = 0.65

    model = _build_dummy_jepa(
        img_size=img_size,
        history_size=history_size,
        action_dim=action_dim,
    )
    planner = CEMPlanner(
        model=model,
        planning_horizon=planning_horizon,
        action_dim=action_dim,
        num_samples=32,
        num_elites=4,
        iterations=2,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    init_pixels = mx.random.normal((1, 1, history_size, 3, img_size, img_size))
    goal_pixels = mx.random.normal((1, 1, 1, 3, img_size, img_size))

    info_dict = {"pixels": init_pixels, "goal": goal_pixels}
    best_actions, _, _ = planner.plan(info_dict)

    assert mx.all(best_actions >= lower_bound).item()
    assert mx.all(best_actions <= upper_bound).item()


def test_planner_input_validation():
    """Verifies that invalid hyperparameters raise ValueError."""
    model = _build_dummy_jepa()

    # planning_horizon <= 0
    with pytest.raises(ValueError, match="planning_horizon must be positive"):
        CEMPlanner(model=model, planning_horizon=0)

    # action_dim <= 0
    with pytest.raises(ValueError, match="action_dim must be positive"):
        CEMPlanner(model=model, action_dim=0)

    # num_samples <= 0
    with pytest.raises(ValueError, match="num_samples must be positive"):
        CEMPlanner(model=model, num_samples=0)

    # num_elites <= 0
    with pytest.raises(ValueError, match="num_elites must be positive"):
        CEMPlanner(model=model, num_elites=0)

    # num_elites > num_samples
    with pytest.raises(ValueError, match="num_elites.*cannot exceed num_samples"):
        CEMPlanner(model=model, num_samples=10, num_elites=12)

    # iterations <= 0
    with pytest.raises(ValueError, match="iterations must be positive"):
        CEMPlanner(model=model, iterations=0)

    # alpha not in [0, 1]
    with pytest.raises(ValueError, match="alpha must be in"):
        CEMPlanner(model=model, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        CEMPlanner(model=model, alpha=1.5)

    # lower_bound > upper_bound
    with pytest.raises(ValueError, match="lower_bound.*cannot be greater than upper_bound"):
        CEMPlanner(model=model, lower_bound=1.0, upper_bound=-1.0)


def test_planner_missing_info_dict_keys():
    """Verifies that missing pixels or goal in info_dict raises an error."""
    model = _build_dummy_jepa()
    planner = CEMPlanner(model=model, planning_horizon=4, action_dim=10)

    with pytest.raises(KeyError, match="pixels"):
        planner.plan({"goal": mx.zeros((1, 1, 1, 3, 96, 96))})

    with pytest.raises(KeyError, match="goal"):
        planner.plan({"pixels": mx.zeros((1, 1, 3, 3, 96, 96))})
