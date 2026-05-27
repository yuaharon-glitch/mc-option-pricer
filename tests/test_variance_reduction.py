"""Tests for quasi_random.py and variance_reduction.py."""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricer.quasi_random import SobolEngine, discrepancy, sobol_to_normal


class TestSobolDiscrepancy:
    """Sobol star discrepancy D*_N < N^{-0.9} for N=1024."""

    def test_star_discrepancy_threshold(self):
        """D*_1024 < 1024^{-0.9} ≈ 0.00186 for 1D Sobol."""
        N = 1024
        engine = SobolEngine(dimension=1, scramble=False, seed=42)
        u = engine.random(N)[:, 0]   # shape (N,)
        D = discrepancy(u)
        threshold = N ** (-0.9)

        assert D < threshold, (
            f"Sobol D*_{N} = {D:.6f} is not < threshold {threshold:.6f} = {N}^{{-0.9}}. "
            f"Implementation may have a bug in direction numbers."
        )

    def test_discrepancy_unscrambled_better_than_pseudorandom(self):
        """Sobol D* should be better (lower) than typical pseudo-random."""
        N = 1024
        engine = SobolEngine(dimension=1, scramble=False, seed=42)
        u_sobol = engine.random(N)[:, 0]
        D_sobol = discrepancy(u_sobol)

        rng = np.random.default_rng(42)
        u_rand = rng.uniform(0, 1, N)
        D_rand = discrepancy(u_rand)

        assert D_sobol < D_rand, (
            f"Sobol D*={D_sobol:.6f} should be < pseudo-random D*={D_rand:.6f}"
        )

    def test_sobol_output_in_unit_interval(self):
        """All Sobol points should be in [0, 1)."""
        engine = SobolEngine(dimension=4, scramble=True, seed=42)
        u = engine.random(512)
        assert (u >= 0.0).all() and (u < 1.0).all()

    def test_sobol_shape(self):
        engine = SobolEngine(dimension=3, scramble=True, seed=42)
        u = engine.random(100)
        assert u.shape == (100, 3)

    def test_sobol_reset(self):
        """reset() produces the same sequence."""
        engine = SobolEngine(dimension=2, scramble=True, seed=42)
        u1 = engine.random(50)
        engine.reset()
        u2 = engine.random(50)
        np.testing.assert_array_equal(u1, u2)

    def test_sobol_sequential_consistency(self):
        """Generating in batches equals generating all at once."""
        engine = SobolEngine(dimension=2, scramble=True, seed=42)
        u_all = engine.random(100)

        engine.reset()
        u_batch1 = engine.random(40)
        u_batch2 = engine.random(60)
        u_concat = np.concatenate([u_batch1, u_batch2], axis=0)

        np.testing.assert_array_equal(u_all, u_concat)

    def test_sobol_to_normal_shape(self):
        engine = SobolEngine(dimension=5, scramble=True, seed=42)
        u = engine.random(200)
        z = sobol_to_normal(u)
        assert z.shape == (200, 5)

    def test_sobol_to_normal_approximately_standard(self):
        """Transformed Sobol samples have approximately zero mean and unit variance."""
        N = 2048
        engine = SobolEngine(dimension=1, scramble=True, seed=42)
        u = engine.random(N)
        z = sobol_to_normal(u)[:, 0]
        assert abs(z.mean()) < 0.1, f"Mean={z.mean():.4f}, expected ~0"
        assert abs(z.std() - 1.0) < 0.1, f"Std={z.std():.4f}, expected ~1"

    def test_1d_dimension_error(self):
        with pytest.raises(ValueError):
            SobolEngine(dimension=0)

    def test_max_dimension(self):
        """Dimension 21 should work without error."""
        engine = SobolEngine(dimension=21, scramble=True, seed=42)
        u = engine.random(32)
        assert u.shape == (32, 21)
        assert (u >= 0).all() and (u < 1).all()
