"""Catenary model: 2D fitting and 3D reconstruction.

Model:  v(u) = v0 + c * cosh((u - u0) / c) - c
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(slots=True)
class CatenaryParams:
    """Fitted catenary parameters (c, u0, v0) and RMSE for one wire."""

    c: float
    u0: float
    v0: float
    rmse: float


def evaluate(u: np.ndarray, c: float, u0: float, v0: float) -> np.ndarray:
    """Evaluate the catenary model at along-wire coordinates ``u``."""
    c = max(float(c), 1e-6)
    return v0 + c * np.cosh((u - u0) / c) - c


def fit(u: np.ndarray, v: np.ndarray) -> CatenaryParams:
    """Fit a catenary to (u, v) data via robust least squares (soft_l1)."""
    if len(u) != len(v):
        raise ValueError("u and v must have the same length.")
    if len(u) < 3:
        raise ValueError("At least 3 points are required to fit a catenary.")

    span = max(float(u.max() - u.min()), 1e-3)

    x0 = np.array([
        max(span / 4.0, 0.5),   # c  — initial guess: gentle sag
        float(np.median(u)),    # u0 — midpoint of the wire
        float(v.min()),         # v0 — lowest observed height
    ])

    bounds = (
        [1e-4,            u.min() - span,   v.min() - span],
        [span * 10 + 1.0, u.max() + span,   v.max() + span],
    )

    result = least_squares(
        lambda p: evaluate(u, *p) - v,
        x0=x0,
        bounds=bounds,
        loss="soft_l1",
    )

    c, u0, v0 = result.x
    fitted_v = evaluate(u, c, u0, v0)
    rmse = float(np.sqrt(np.mean((fitted_v - v) ** 2)))

    return CatenaryParams(c=float(c), u0=float(u0), v0=float(v0), rmse=rmse)


def to_3d(
    params: CatenaryParams,
    centroid: np.ndarray,
    wire_dir: np.ndarray,
    up: np.ndarray,
    u_min: float,
    u_max: float,
    n_points: int = 200,
) -> np.ndarray:
    """Sample the fitted catenary back into 3D world coordinates."""
    u_grid = np.linspace(u_min, u_max, n_points)
    v_grid = evaluate(u_grid, params.c, params.u0, params.v0)
    return centroid + np.outer(u_grid, wire_dir) + np.outer(v_grid, up)