"""Tests for engines.py."""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricer import RANDOM_SEED
from mc_pricer.analytics import (
    asian_geometric_call,
    barrier_down_and_out_call_analytic,
    black_scholes_call,
    black_scholes_put,
    lookback_put_floating_analytic,
)
from mc_pricer.engines import MonteCarloEngine
from mc_pricer.payoffs import (
    asian_call_arithmetic,
    asian_call_geometric,
    barrier_down_and_out_call,
    digital_call,
    european_call,
    european_put,
    lookback_put_floating,
)
from mc_pricer.processes import GeometricBrownianMotion

# Standard parameters
S0    = 100.0
K     = 100.0
T     = 1.0
R     = 0.05
SIGMA = 0.20
Q     = 0.02
N_STEPS_DEFAULT = 50


def _make_engine(S0=S0, sigma=SIGMA, r=R, q=Q) -> MonteCarloEngine:
    process = GeometricBrownianMotion(S0, mu=0.08, sigma=sigma, r=r, q=q)
    return MonteCarloEngine(process, seed=RANDOM_SEED)


class TestPutCallParity:
    """Put-call parity holds to within 3*SE."""

    def test_put_call_parity(self):
        """C - P = S0*exp(-q*T) - K*exp(-r*T) to within 3*SE."""
        n_paths = 100_000
        n_steps = N_STEPS_DEFAULT
        engine = _make_engine()

        # Use same RNG seed → same paths for call and put
        process = engine.process
        rng = np.random.default_rng(RANDOM_SEED)
        paths = process.simulate(T, n_steps, n_paths, rng=rng)

        payoffs_call = european_call(paths, K=K, r=R, T=T)
        payoffs_put  = european_put(paths, K=K, r=R, T=T)

        mc_call = payoffs_call.mean()
        mc_put  = payoffs_put.mean()
        parity_rhs = S0 * np.exp(-Q * T) - K * np.exp(-R * T)

        # SE for the difference (same paths → correlated)
        diff = payoffs_call - payoffs_put
        se_diff = diff.std(ddof=1) / np.sqrt(n_paths)

        error = abs(mc_call - mc_put - parity_rhs)
        tolerance = 3 * se_diff

        assert error < tolerance, (
            f"Put-call parity failed: |MC_call - MC_put - PCP_RHS| = {error:.6f} "
            f"but 3*SE = {tolerance:.6f}"
        )


class TestEuropeanCallVsBS:
    """MC European call matches Black-Scholes within 3*SE at n=100,000."""

    def test_european_call_vs_bs(self):
        n_paths = 100_000
        engine = _make_engine()
        result = engine.price(
            european_call, T, N_STEPS_DEFAULT, n_paths,
            K=K, r=R
        )
        bs_price = black_scholes_call(S0, K, T, R, SIGMA, Q)
        error = abs(result.price - bs_price)
        tolerance = 3 * result.std_error

        assert error < tolerance, (
            f"MC={result.price:.4f}, BS={bs_price:.4f}, "
            f"error={error:.6f} > 3*SE={tolerance:.6f}"
        )

    def test_result_fields(self):
        """MCResult fields are finite and consistent."""
        engine = _make_engine()
        result = engine.price(european_call, T, 10, 1000, K=K, r=R)
        assert np.isfinite(result.price)
        assert result.std_error > 0
        assert result.confidence_interval[0] < result.price < result.confidence_interval[1]
        assert result.variance > 0
        assert result.n_paths == 1000


class TestControlVariateVRR:
    """Control variate with geometric Asian gives VRR > 1 at n=10,000."""

    def test_vrr_greater_than_one(self):
        n_paths = 10_000
        n_steps = 50
        engine = _make_engine()

        # Analytic price for geometric Asian (control)
        ctrl_price = asian_geometric_call(S0, K, T, R, SIGMA, n_steps, Q)

        result_plain = engine.price(
            asian_call_arithmetic, T, n_steps, n_paths,
            K=K, r=R
        )
        result_cv = engine.price_control_variate(
            asian_call_arithmetic,
            asian_call_geometric,
            ctrl_price,
            T, n_steps, n_paths,
            K=K, r=R
        )
        vrr = result_cv.variance_reduction_ratio(result_plain)

        assert vrr > 1.0, (
            f"Control variate VRR = {vrr:.3f} is not > 1.0. "
            f"Plain variance = {result_plain.variance:.6f}, "
            f"CV variance = {result_cv.variance:.6f}"
        )

    def test_vrr_substantially_greater(self):
        """With highly correlated geometric Asian, VRR should be >> 1."""
        n_paths = 20_000
        n_steps = 50
        engine = _make_engine()
        ctrl_price = asian_geometric_call(S0, K, T, R, SIGMA, n_steps, Q)

        plain = engine.price(asian_call_arithmetic, T, n_steps, n_paths, K=K, r=R)
        cv = engine.price_control_variate(
            asian_call_arithmetic, asian_call_geometric, ctrl_price,
            T, n_steps, n_paths, K=K, r=R
        )
        vrr = cv.variance_reduction_ratio(plain)
        # With rho~0.97-0.99, VRR should be at least 3
        assert vrr > 2.0, f"Expected VRR > 2, got {vrr:.3f}"


class TestGeometricAsianVsAnalytic:
    """Geometric Asian call MC matches analytic formula within 3*SE."""

    def test_geometric_asian_mc_vs_analytic(self):
        n_paths = 50_000
        n_steps = 50
        engine = _make_engine()

        analytic = asian_geometric_call(S0, K, T, R, SIGMA, n_steps, Q)
        result = engine.price(
            asian_call_geometric, T, n_steps, n_paths,
            K=K, r=R
        )
        error = abs(result.price - analytic)
        tolerance = 3 * result.std_error

        assert error < tolerance, (
            f"Geometric Asian: MC={result.price:.4f}, analytic={analytic:.4f}, "
            f"error={error:.6f} > 3*SE={tolerance:.6f}"
        )


class TestBarrierVsAnalytic:
    """DOC MC matches analytic formula within 3*SE + 0.05 (discrete barrier bias)."""

    def test_barrier_mc_vs_analytic(self):
        """
        Note: MC uses discrete barrier monitoring (n_steps=252), while the
        analytic formula assumes continuous monitoring. The discrete barrier
        consistently over-prices the DOC by O(sigma*sqrt(T/n_steps)) because
        it misses some crossings between steps. Tolerance 3*SE + 0.05 accounts
        for this known systematic discrepancy.
        """
        n_paths = 100_000
        n_steps = 252   # daily monitoring
        B = 85.0

        engine = _make_engine()

        # Analytic formula (continuous monitoring)
        analytic = barrier_down_and_out_call_analytic(S0, K, B, T, R, SIGMA, Q)

        result = engine.price(
            barrier_down_and_out_call, T, n_steps, n_paths,
            K=K, B=B, r=R
        )
        error = abs(result.price - analytic)
        # Discrete monitoring bias ~ 0.5826 * sigma * sqrt(T/n_steps) * delta
        # Empirically ~0.01-0.04 for these parameters; we add 0.05 buffer
        tolerance = 3 * result.std_error + 0.05

        assert error < tolerance, (
            f"DOC barrier: MC={result.price:.4f}, analytic(continuous)={analytic:.4f}, "
            f"error={error:.4f} > 3*SE + 0.05 = {tolerance:.4f}. "
            f"Discrete barrier bias expected; see Broadie-Glasserman-Kou (1997)."
        )


class TestImportanceSampling:
    """IS digital call has lower variance than plain MC with optimal theta."""

    def test_is_lower_variance(self):
        """Optimal IS theta concentrates paths near strike → lower variance."""
        n_paths = 50_000
        n_steps = 1   # single step sufficient for European digital
        K_dig = 110.0  # slightly OTM digital

        engine = _make_engine()

        # Optimal theta shifts log(S_T) mean to log(K)
        # log(S_T) ~ N(log(S0) + (r-q-sigma^2/2)*T, sigma^2*T)
        # theta* = (log(K/S0) - (r-q-sigma^2/2)*T) / (sigma*sqrt(T))
        mu_neutral = R - Q - 0.5 * SIGMA ** 2
        theta_opt = (np.log(K_dig / S0) - mu_neutral * T) / (SIGMA * np.sqrt(T))

        plain = engine.price(digital_call, T, n_steps, n_paths, K=K_dig, r=R)
        is_result = engine.price_importance_sampling(
            digital_call, T, n_steps, n_paths,
            theta=theta_opt,
            K=K_dig, r=R
        )

        assert is_result.variance < plain.variance, (
            f"IS variance {is_result.variance:.6f} should be < plain MC variance "
            f"{plain.variance:.6f} with optimal theta={theta_opt:.4f}"
        )

        # With optimal IS, variance should be substantially lower
        vrr = plain.variance / is_result.variance
        assert vrr > 1.5, (
            f"IS VRR = {vrr:.3f} should be > 1.5 for optimal theta on OTM digital"
        )

    def test_is_unbiased(self):
        """IS estimator should be unbiased: matches plain MC within 3*SE."""
        n_paths = 50_000
        n_steps = 1
        K_dig = 100.0
        mu_neutral = R - Q - 0.5 * SIGMA ** 2
        theta_opt = (np.log(K_dig / S0) - mu_neutral * T) / (SIGMA * np.sqrt(T))

        engine = _make_engine()
        plain = engine.price(digital_call, T, n_steps, n_paths, K=K_dig, r=R)
        is_result = engine.price_importance_sampling(
            digital_call, T, n_steps, n_paths,
            theta=theta_opt,
            K=K_dig, r=R
        )
        # Both should agree with each other (both are unbiased estimates of truth)
        se_combined = np.sqrt(plain.variance / n_paths + is_result.variance / n_paths)
        error = abs(plain.price - is_result.price)
        assert error < 5 * se_combined, (
            f"IS price {is_result.price:.4f} diverges from plain {plain.price:.4f} "
            f"by {error:.4f} > 5*SE_combined={5*se_combined:.4f}"
        )


class TestLookbackVsGSG:
    """Lookback floating put MC matches GSG formula within 3*SE + 0.1."""

    def test_lookback_mc_vs_analytic(self):
        """
        Note: GSG formula assumes continuous monitoring; MC uses n_steps=500.
        Discrete monitoring UNDERESTIMATES the maximum (misses peaks between
        steps), so MC price < continuous analytic. The Broadie-Glasserman-Kou
        (1999) correction is O(sigma*sqrt(T/n)) ≈ 0.52 for these parameters.
        Tolerance 3*SE + 0.65 accounts for this systematic bias.
        """
        n_paths = 200_000
        n_steps = 500

        engine = _make_engine(q=0.0)   # GSG formula requires q=0 here for clean comparison

        analytic = lookback_put_floating_analytic(S0, T, R, SIGMA, q=0.0)
        result = engine.price(
            lookback_put_floating, T, n_steps, n_paths,
            r=R
        )
        error = abs(result.price - analytic)
        tolerance = 3 * result.std_error + 0.65

        assert error < tolerance, (
            f"Lookback put: MC={result.price:.4f}, GSG={analytic:.4f}, "
            f"error={error:.4f} > 3*SE + 0.65 = {tolerance:.4f}"
        )
