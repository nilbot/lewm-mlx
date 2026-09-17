"""Unit tests for shared ImageNet pixel preprocessing."""

import numpy as np

import mlx.core as mx

from lewm_mlx.preprocessing import IMAGENET_MEAN, IMAGENET_STD, imagenet_normalize


def test_imagenet_normalize_matches_reference_formula():
    """Verifies channel-wise standardization against the explicit formula."""
    pixels = mx.random.uniform(shape=(2, 3, 3, 8, 8))
    out = np.array(imagenet_normalize(pixels))

    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3, 1, 1)
    expected = (np.array(pixels) - mean) / std

    assert out.shape == pixels.shape
    assert np.allclose(out, expected, atol=1e-6)


def test_imagenet_normalize_broadcasts_over_4d_input():
    """Verifies channel-axis broadcasting for [B, C, H, W] frame tensors."""
    pixels = mx.zeros(shape=(1, 3, 4, 4))
    out = np.array(mx.mean(imagenet_normalize(pixels), axis=(0, 2, 3)))

    expected = -np.array(IMAGENET_MEAN) / np.array(IMAGENET_STD)
    assert np.allclose(out, expected, atol=1e-6)
