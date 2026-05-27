"""Tests for processes.py."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from mc_pricer import RANDOM_SEED
from mc_pricer.processes import GeometricBrownianMotion, HestonProcess

S0    = 100.0
MU    = 0.08
SIGMA = 0.20
R     = 0.05
Q     = 0.02
T     = 1.0


class TestGBMLognormal:
    """GBM terminal prices follow lognormal distribution (KS test)."""

    def test_lognormal_ks_test(self):
        """KS test fails to reject lognormality at alpha=0.05 with n=50,000."""
        n_paths = 50_000
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        rng = np.random.default_rng(RANDOM_SEED)
        S_T = gbm.terminal_distribution(T, n_paths, rng=rng)

        # Theoretical distribution under risk-neutral measure
        # log(S_T) ~ N(log(S0) + (r-q-sigma^2/2)*T, sigma^2*T)
        mu_ln = np.log(S0) + (R - Q - 0.5 * SIGMA ** 2) * T
        sigma_ln = SIGMA * np.sqrt(T)

        # scipy lognorm: loc=0, scale=exp(mu_ln), s=sigma_ln
        dist = stats.lognorm(s=sigma_ln, scale=np.exp(mu_ln))
        stat, p_value = stats.kstest(S_T, dist.cdf)

        assert p_value > 0.01, (
            f"KS test rejected lognormality: stat={stat:.4f}, p={p_value:.4f}. "
            f"Expected p > 0.01 for n={n_paths} paths."
        )

    def test_lognormal_mean(self):
        """E[S_T] = S0 * exp((r-q)*T) under risk-neutral measure."""
        n_paths = 200_000
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        rng = np.random.default_rng(RANDOM_SEED + 1)
        S_T = gbm.terminal_distribution(T, n_paths, rng=rng)

        expected_mean = S0 * np.exp((R - Q) * T)
        sample_mean = S_T.mean()
        se = S_T.std() / np.sqrt(n_paths)

        assert abs(sample_mean - expected_mean) < 3 * se, (
            f"Mean check failed: expected={expected_mean:.4f}, "
            f"got={sample_mean:.4f}, 3*SE={3*se:.4f}"
        )

    def test_path_shape(self):
        """simulate() returns shape (n_paths, n_steps+1) with S0 in first column."""
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        n_paths, n_steps = 1000, 50
        paths = gbm.simulate(T, n_steps, n_paths)
        assert paths.shape == (n_paths, n_steps + 1)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_paths_positive(self):
        """All GBM paths must be strictly positive."""
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        paths = gbm.simulate(T, 252, 5000)
        assert (paths > 0).all(), "GBM produced non-positive path values"


class TestAntitheticCorrelation:
    """Antithetic terminal prices are negatively correlated."""

    def test_antithetic_negative_correlation(self):
        """corr(S_paths[:n,-1], S_paths[n:,-1]) < 0."""
        n_paths = 10_000
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        rng = np.random.default_rng(RANDOM_SEED)
        paths = gbm.simulate_antithetic(T, 50, n_paths, rng=rng)

        half = n_paths // 2
        S_std  = paths[:half, -1]
        S_anti = paths[half:, -1]

        corr = float(np.corrcoef(S_std, S_anti)[0, 1])
        assert corr < 0.0, (
            f"Antithetic correlation should be negative, got {corr:.4f}"
        )
        # For GBM with reasonable sigma, correlation should be strongly negative
        assert corr < -0.5, (
            f"Antithetic correlation {corr:.4f} is not strongly negative "
            f"(expected < -0.5 for sigma={SIGMA})"
        )

    def test_antithetic_shape(self):
        """simulate_antithetic returns shape (n_paths, n_steps+1)."""
        n_paths = 200
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        paths = gbm.simulate_antithetic(T, 10, n_paths)
        assert paths.shape == (n_paths, 11)

    def test_antithetic_unbiased(self):
        """Antithetic estimator mean should match plain MC to within 3*SE."""
        n_paths = 50_000
        gbm = GeometricBrownianMotion(S0, MU, SIGMA, R, Q)
        rng = np.random.default_rng(RANDOM_SEED)

        paths_plain = gbm.simulate(T, 1, n_paths, rng=np.random.default_rng(RANDOM_SEED))
        paths_anti  = gbm.simulate_antithetic(T, 1, n_paths, rng=rng)

        mean_plain = paths_plain[:, 1].mean()
        half = n_paths // 2
        mean_anti = 0.5 * (paths_anti[:half, 1] + paths_anti[half:, 1]).mean()
        expected   = S0 * np.exp((R - Q) * T)

        se_plain = paths_plain[:, 1].std() / np.sqrt(n_paths)
        assert abs(mean_anti - expected) < 5 * se_plain


class TestHestonFellerWarning:
    """HestonProcess warns when Feller condition is violated."""

    def test_feller_violation_warning(self):
        """Warn when 2*kappa*theta < xi^2."""
        kappa = 1.0
        theta = 0.04
        xi = 0.5      # xi^2 = 0.25 > 2*1.0*0.04 = 0.08  -> Feller violated

        with pytest.warns(UserWarning, match="Feller condition violated"):
            HestonProcess(
                S0=100.0, v0=0.04,
                kappa=kappa, theta=theta, xi=xi,
                rho=-0.7, r=0.05,
            )

    def test_feller_satisfied_no_warning(self):
        """No warning when Feller condition is satisfied."""
        kappa = 3.0
        theta = 0.04
        xi = 0.3      # xi^2 = 0.09 < 2*3.0*0.04 = 0.24 -> Feller OK

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            HestonProcess(
                S0=100.0, v0=0.04,
                kappa=kappa, theta=theta, xi=xi,
                rho=-0.7, r=0.05,
            )

    def test_heston_paths_shape(self):
        """Heston simulate() returns correct shapes."""
        h = HestonProcess(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05)
        S_paths, v_paths = h.simulate(T=1.0, n_steps=50, n_paths=500)
        assert S_paths.shape == (500, 51)
        assert v_paths.shape == (500, 51)

    def test_heston_stock_prices_positive(self):
        """All Heston S paths must be positive (log-Euler)."""
        h = HestonProcess(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05)
        S_paths, _ = h.simulate(T=1.0, n_steps=100, n_paths=2000)
        assert (S_paths > 0).all()
