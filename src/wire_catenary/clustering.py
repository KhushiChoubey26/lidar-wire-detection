"""DBSCAN wire clustering in the (x, y) plane with auto epsilon."""

from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def _estimate_eps(xy: np.ndarray, k: int = 12, quantile: float = 0.90) -> float:
    """Pick DBSCAN eps from the 90th-percentile of k-NN distances."""
    n = len(xy)
    if n == 0:
        return 1.0

    if n <= k:
        return max(1.0, float(np.ptp(xy, axis=0).mean()) / 10.0)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(xy)
    distances = nbrs.kneighbors(xy)[0]
    kth_distances = distances[:, -1]
    return float(max(np.quantile(kth_distances, quantile), 0.25))


def cluster_wires(
    points: np.ndarray,
    min_samples: int = 15,
    min_cluster_size: int = 40,
) -> List[Tuple[int, np.ndarray]]:
    """Cluster wires via DBSCAN on (x, y) with auto eps.

    Falls back to relaxed parameters if the first pass finds nothing.
    Returns (label, points) tuples for clusters above min_cluster_size.
    """
    if len(points) == 0:
        return []

    xy = points[:, :2]
    eps = _estimate_eps(xy, k=min(12, max(1, len(xy) - 1)))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)

    clusters = _collect(points, labels, min_cluster_size)

    if not clusters:
        relaxed_eps = max(eps * 1.5, 0.5)
        relaxed_min_samples = max(8, min_samples // 2)
        relaxed_min_cluster_size = max(20, min_cluster_size // 2)

        labels = DBSCAN(
            eps=relaxed_eps,
            min_samples=relaxed_min_samples,
        ).fit_predict(xy)
        clusters = _collect(points, labels, relaxed_min_cluster_size)

    return clusters


def _collect(
    points: np.ndarray,
    labels: np.ndarray,
    min_size: int,
) -> List[Tuple[int, np.ndarray]]:
    """Collect non-noise clusters above a minimum size threshold."""
    return [
        (int(lbl), points[labels == lbl])
        for lbl in sorted(set(labels))
        if lbl != -1 and int((labels == lbl).sum()) >= min_size
    ]