"""End-to-end pipeline: load, preprocess, cluster, fit, report."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .catenary import CatenaryParams, fit as fit_catenary, to_3d
from .clustering import cluster_wires
from .io import load_points
from .plane_fit import fit_plane, project
from .preprocessing import downsample, remove_outliers


@dataclass(slots=True)
class WireResult:
    """Fitting result for one detected wire."""

    wire_id: int
    n_points: int
    centroid: np.ndarray
    wire_dir: np.ndarray
    up: np.ndarray
    params: CatenaryParams
    u_min: float
    u_max: float
    curve_3d: np.ndarray
    cluster_points: np.ndarray

    def __repr__(self) -> str:
        p = self.params
        return (
            f"WireResult(id={self.wire_id}, n={self.n_points}, "
            f"c={p.c:.3f}, u0={p.u0:.3f}, v0={p.v0:.3f}, rmse={p.rmse:.4f})"
        )


def run_dataset(
    path: Path | str,
    *,
    max_points: int = 50_000,
    z_outlier_thresh: float = 3.5,
    min_cluster_samples: int = 15,
    min_cluster_size: int = 40,
) -> List[WireResult]:
    """Run the full detection + fitting pipeline on one parquet file.

    Returns a list of WireResult sorted by RMSE (best fit first).
    Clusters that fail to fit are skipped with a warning.
    """
    points = load_points(path)
    points = remove_outliers(points, z_thresh=z_outlier_thresh)
    points = downsample(points, max_points=max_points)

    clusters = cluster_wires(
        points,
        min_samples=min_cluster_samples,
        min_cluster_size=min_cluster_size,
    )

    if not clusters:
        return []

    results: List[WireResult] = []

    for label, cluster_pts in clusters:
        try:
            if len(cluster_pts) < 3:
                continue

            centroid, wire_dir, up = fit_plane(cluster_pts)
            u, v = project(cluster_pts, centroid, wire_dir, up)

            if len(u) < 3:
                continue

            u_min = float(np.min(u))
            u_max = float(np.max(u))

            if np.isclose(u_min, u_max):
                continue

            params = fit_catenary(u, v)
            curve = to_3d(params, centroid, wire_dir, up, u_min, u_max)

            results.append(
                WireResult(
                    wire_id=label,
                    n_points=len(cluster_pts),
                    centroid=centroid,
                    wire_dir=wire_dir,
                    up=up,
                    params=params,
                    u_min=u_min,
                    u_max=u_max,
                    curve_3d=curve,
                    cluster_points=cluster_pts,
                )
            )
        except Exception as exc:
            warnings.warn(f"Skipping cluster {label}: {exc}")
            continue

    results.sort(key=lambda r: r.params.rmse)
    return results