"""3D scatter + 2D catenary profile plots."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .catenary import evaluate
from .pipeline import WireResult
from .plane_fit import project


def plot_results(
    points: np.ndarray,
    results: List[WireResult],
    title: str = "",
    show: bool = True,
) -> plt.Figure:
    """3D point cloud with fitted curves + one 2D catenary profile per wire."""
    if not results:
        fig = plt.figure(figsize=(10, 6))
        ax3d = fig.add_subplot(111, projection="3d")
        ax3d.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=4, alpha=0.20, color="gray", label="LiDAR points",
        )
        ax3d.set_title(f"{title} — no fitted wires" if title else "No fitted wires")
        ax3d.set_xlabel("x (m)")
        ax3d.set_ylabel("y (m)")
        ax3d.set_zlabel("z (m)")
        if show:
            plt.show()
        return fig

    n = len(results)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(1, n))
    fig = plt.figure(figsize=(14, 6 * (1 + n)))

    ax3d = fig.add_subplot(n + 1, 1, 1, projection="3d")
    ax3d.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=4, alpha=0.20, color="gray", label="LiDAR points",
    )

    for i, r in enumerate(results):
        curve = r.curve_3d
        ax3d.plot(
            curve[:, 0], curve[:, 1], curve[:, 2],
            color=cmap(i), linewidth=2.5, label=f"wire {r.wire_id}",
        )

    ax3d.set_title(f"{title} — 3D view" if title else "3D view")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="best")

    for i, r in enumerate(results):
        u_all, v_all = project(r.cluster_points, r.centroid, r.wire_dir, r.up)
        u_fit = np.linspace(r.u_min, r.u_max, 300)
        v_fit = evaluate(u_fit, r.params.c, r.params.u0, r.params.v0)

        ax = fig.add_subplot(n + 1, 1, i + 2)
        ax.scatter(
            u_all, v_all,
            s=6,
            alpha=0.25,
            color="gray",
            label="cluster points",
        )
        ax.plot(
            u_fit,
            v_fit,
            color=cmap(i),
            linewidth=2,
            label=f"catenary  c={r.params.c:.2f}  RMSE={r.params.rmse:.4f} m",
        )
        ax.set_title(f"Wire {r.wire_id} — 2D catenary profile in wire plane")
        ax.set_xlabel("u (along-wire distance, m)")
        ax.set_ylabel("v (height in wire plane, m)")
        ax.legend(loc="best")

    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return fig