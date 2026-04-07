"""Unit tests for wire_catenary.catenary."""

import numpy as np
import pytest

from wire_catenary.catenary import CatenaryParams, evaluate, fit


class TestEvaluate:
    def test_trough_at_u0(self):
        """Minimum value should occur exactly at u0."""
        c, u0, v0 = 10.0, 5.0, 2.0
        u = np.linspace(0, 10, 1000)
        v = evaluate(u, c, u0, v0)
        assert pytest.approx(float(u[v.argmin()]), abs=0.02) == u0

    def test_trough_value_equals_v0(self):
        """Value at the trough should equal v0."""
        c, u0, v0 = 8.0, 3.0, -1.0
        assert pytest.approx(evaluate(np.array([u0]), c, u0, v0)[0], abs=1e-9) == v0

    def test_symmetry(self):
        """Catenary is symmetric around u0."""
        c, u0, v0 = 5.0, 0.0, 0.0
        u_pos = np.array([1.0, 2.0, 3.0])
        u_neg = -u_pos
        np.testing.assert_allclose(
            evaluate(u_pos, c, u0, v0),
            evaluate(u_neg, c, u0, v0),
            rtol=1e-10,
        )

    def test_clamps_small_c(self):
        """c < 1e-6 should be clamped, not raise."""
        v = evaluate(np.array([0.0]), c=0.0, u0=0.0, v0=0.0)
        assert np.isfinite(v[0])


class TestFit:
    def _make_data(self, c=10.0, u0=2.0, v0=0.5, noise=0.02):
        rng = np.random.default_rng(42)
        u = np.linspace(-10, 10, 300)
        v = evaluate(u, c, u0, v0) + rng.normal(0, noise, size=len(u))
        return u, v, c, u0, v0

    def test_recovers_parameters(self):
        u, v, true_c, true_u0, true_v0 = self._make_data()
        params = fit(u, v)
        assert pytest.approx(params.c, rel=0.05) == true_c
        assert pytest.approx(params.u0, abs=0.2) == true_u0
        assert pytest.approx(params.v0, abs=0.1) == true_v0

    def test_rmse_low_for_clean_data(self):
        u, v, *_ = self._make_data(noise=0.01)
        params = fit(u, v)
        assert params.rmse < 0.05

    def test_returns_dataclass(self):
        u = np.linspace(-5, 5, 50)
        v = evaluate(u, 5.0, 0.0, 0.0)
        params = fit(u, v)
        assert isinstance(params, CatenaryParams)
        assert hasattr(params, "rmse")

    def test_fit_raises_on_length_mismatch(self):
        u = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="same length"):
            fit(u, v)

    def test_fit_raises_on_too_few_points(self):
        u = np.array([0.0, 1.0])
        v = np.array([0.0, 0.1])
        with pytest.raises(ValueError, match="At least 3 points"):
            fit(u, v)