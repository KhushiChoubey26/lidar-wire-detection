"""Outlier removal and downsampling."""

import numpy as np


def remove_outliers(points: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    """Drop points whose z-score on any axis exceeds *z_thresh*."""
    if len(points) == 0:
        return points

    mean = points.mean(axis=0)
    std = points.std(axis=0)

    # Avoid division by very small values
    std[std < 1e-9] = 1.0

    z_scores = np.abs((points - mean) / std)
    mask = np.all(z_scores < z_thresh, axis=1)
    return points[mask]


def downsample(points: np.ndarray, max_points: int = 20_000) -> np.ndarray:
    """Randomly subsample to at most *max_points* rows."""
    if len(points) <= max_points:
        return points

    rng = np.random.default_rng(seed=0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]