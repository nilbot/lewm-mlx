"""Shared pixel preprocessing for the MLX Le World Model entry points.

Training (:mod:`lewm_mlx.train`) and inference (:mod:`demo`) must standardize
frames identically so the visual encoder always receives inputs drawn from the
distribution it was trained on. The statistics mirror the PyTorch reference
(``le-wm-ref/utils.py``, ``stable_pretraining.data.dataset_stats.ImageNet``).
"""

from typing import Tuple

import mlx.core as mx

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def imagenet_normalize(pixels: mx.array) -> mx.array:
    r"""Standardizes channel-first frame tensors with ImageNet channel statistics.

    Computes, per color channel :math:`c \in \{R, G, B\}`:

    .. math::
        \hat{\mathbf{x}}_c = \frac{\mathbf{x}_c - \mu_c}{\sigma_c},
        \qquad \boldsymbol{\mu} = (0.485, 0.456, 0.406),
        \qquad \boldsymbol{\sigma} = (0.229, 0.224, 0.225)

    Args:
        pixels: Frame tensor whose channel axis is at position ``-3`` and whose
            values lie in :math:`[0, 1]`; e.g. :math:`[B, T, C, H, W]` or
            :math:`[B, C, H, W]`.

    Returns:
        Standardized tensor with the same shape as ``pixels``.
    """
    broadcast_shape = (1,) * (pixels.ndim - 3) + (3, 1, 1)
    mean = mx.array(IMAGENET_MEAN).reshape(broadcast_shape)
    std = mx.array(IMAGENET_STD).reshape(broadcast_shape)
    return (pixels - mean) / std
