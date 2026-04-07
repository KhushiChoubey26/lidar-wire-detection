"""Unit tests for wire_catenary.plane_fit."""

import numpy as np
import pytest

from wire_catenary.plane_fit import fit_plane, project


def _make_wire(direction=(1, 0, 0), length=20.0, sag=0.5, noise=0.01, n=200):
    """Generate synthetic wire points along a given direction with vertical sag."""
    rng = np.random.default_rng(0)
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    t = np.linspace(-length / 2, length / 2, n)

    points = np.outer(t, d)
    points[:, 2] += sag * np.cosh(t / (length / 4)) - sag
    points += rng.normal(0, noise, size=points.shape)
    return points


class TestFitPlane:
    def test_wire_dir_aligned_with_true_direction(self):
        """wire_dir should be nearly parallel to the true wire direction."""
        true_dir = np.array([1.0, 0.2, 0.0])
        true_dir /= np.linalg.norm(true_dir)
        points = _make_wire(direction=true_dir)
        _, wire_dir, _ = fit_plane(points)
        cos_sim = abs(np.dot(wire_dir, true_dir))
        assert cos_sim > 0.99

    def test_centroid_near_mean(self):
        points = _make_wire()
        centroid, _, _ = fit_plane(points)
        np.testing.assert_allclose(centroid, points.mean(axis=0), atol=1e-10)

    def test_axes_orthonormal(self):
        points = _make_wire(direction=(0.6, 0.8, 0.0))
        _, wire_dir, up = fit_plane(points)
        assert pytest.approx(np.linalg.norm(wire_dir), abs=1e-9) == 1.0
        assert pytest.approx(np.linalg.norm(up), abs=1e-9) == 1.0
        assert pytest.approx(float(np.dot(wire_dir, up)), abs=1e-6) == 0.0

    def test_near_vertical_direction_does_not_fail(self):
        """Fallback frame construction should handle near-vertical wires safely."""
        points = _make_wire(direction=(0.01, 0.0, 1.0), length=10.0, sag=0.1)
        centroid, wire_dir, up = fit_plane(points)
        assert np.isfinite(centroid).all()
        assert np.isfinite(wire_dir).all()
        assert np.isfinite(up).all()
        assert pytest.approx(np.linalg.norm(wire_dir), abs=1e-9) == 1.0
        assert pytest.approx(np.linalg.norm(up), abs=1e-9) == 1.0


class TestProject:
    def test_u_spans_wire_length(self):
        points = _make_wire(length=20.0)
        centroid, wire_dir, up = fit_plane(points)
        u, _ = project(points, centroid, wire_dir, up)
        assert (u.max() - u.min()) == pytest.approx(20.0, abs=0.5)

    def test_v_captures_sag(self):
        """Projected v values should show a sag pattern (min near centre)."""
        points = _make_wire(length=20.0, sag=1.0)
        centroid, wire_dir, up = fit_plane(points)
        u, v = project(points, centroid, wire_dir, up)
        min_idx = v.argmin()
        assert abs(u[min_idx]) < 3.0