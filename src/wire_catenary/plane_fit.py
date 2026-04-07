"""PCA-based local coordinate frame for catenary fitting."""

from typing import Tuple

import numpy as np


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a wire-aligned (centroid, wire_dir, up) frame via SVD.

    wire_dir is the dominant PCA axis (along the wire).
    up is the in-plane vertical axis (sag direction), orthogonal to wire_dir.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)

    wire_dir = _unit(vh[0])
    global_up = np.array([0.0, 0.0, 1.0])

    # Construct the normal of the vertical plane containing the wire direction.
    normal = np.cross(wire_dir, global_up)
    if np.linalg.norm(normal) < 1e-8:
        normal = np.cross(wire_dir, np.array([1.0, 0.0, 0.0]))
    normal = _unit(normal)

    # In-plane vertical direction (sag axis), orthogonal to wire_dir.
    up = global_up - np.dot(global_up, wire_dir) * wire_dir
    if np.linalg.norm(up) < 1e-8:
        up = np.cross(normal, wire_dir)
    up = _unit(up)

    return centroid, wire_dir, up


def project(
    points: np.ndarray,
    centroid: np.ndarray,
    wire_dir: np.ndarray,
    up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points into local (u, v) wire-plane coordinates."""
    delta = points - centroid

    # Guard against degenerate axes that can arise from near-planar clusters
    if not (np.all(np.isfinite(wire_dir)) and np.all(np.isfinite(up))):
        return np.zeros(len(points)), np.zeros(len(points))

    # Suppress numpy warnings for edge-case overflows; results are checked below
    with np.errstate(all="ignore"):
        u = delta @ wire_dir
        v = delta @ up

    # Replace any non-finite values that slip through numerical edge cases
    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)
    return u, v


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize to unit length; raises if near-zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / norm