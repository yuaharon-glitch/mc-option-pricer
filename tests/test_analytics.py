"""Tests for analytics.py."""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricer.analytics import (
    asian_geometric_call,
    barrier_down_and_out_call_analytic,
    black_scholes_call,
    black_scholes_digital_call,
    black_scholes_put,
    greeks,
    implied_volatility,
    lookback_put_floating_analytic,
)

# Standard test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
SIGMA = 0.20
Q = 0.02


class TestImpliedVolatilityRoundTrip:
    """IV(BS_call(sigma)) recovers sigma to 1e-6."""

    @pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20, 0.30, 0.50, 0.80])
    def test_call_round_trip(self, sigma):
        price = black_scholes_call(S0, K, T, r, sigma, Q)
        sigma_recovered = implied_volatility(price, S0, K, T, r, option_type="call", q=Q)
        assert abs(sigma_recovered - sigma) < 1e-6, (
            f"IV round-trip failed: input sigma={sigma:.4f}, recovered={sigma_recovered:.8f}, "
            f"error={abs(sigma_recovered - sigma):.2e}"
        )

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.40])
    def test_put_round_trip(self, sigma):
        price = black_scholes_put(S0, K, T, r, sigma, Q)
        sigma_recovered = implied_volatility(price, S0, K, T, r, option_type="put", q=Q)
        assert abs(sigma_recovered - sigma) < 1e-6, (
            f"Put IV round-trip failed: sigma={sigma}, recovered={sigma_recovered:.8f}"
        )

    def test_otm_call_round_trip(self):
        """IV round-trip for OTM call (K=120)."""
        K_otm = 120.0
        sigma = 0.25
        price = black_scholes_call(S0, K_otm, T, r, sigma, Q)
        sigma_recovered = implied_volatility(price, S0, K_otm, T, r, option_type="call", q=Q)
        assert abs(sigma_recovered - sigma) < 1e-6

    def test_itm_call_round_trip(self):
        """IV round-trip for ITM call (K=80)."""
        K_itm = 80.0
        sigma = 0.15
        price = black_scholes_call(S0, K_itm, T, r, sigma, Q)
        sigma_recovered = implied_volatility(price, S0, K_itm, T, r, option_type="call", q=Q)
        assert abs(sigma_recovered - sigma) < 1e-6


class TestGreeks:
    """Greeks sanity checks and finite-difference verification to 1e-4."""

    def setup_method(self):
        self.g = greeks(S0, K, T, r, SIGMA, Q)

    # -- Sanity checks

    def test_delta_range(self):
        """Delta for a call must be in (0, 1)."""
        assert 0.0 < self.g["Delta"] < 1.0, f"Delta={self.g['Delta']:.6f} out of (0,1)"

    def test_gamma_positive(self):
        """Gamma must be positive (option price is convex in S)."""
        assert self.g["Gamma"] > 0.0, f"Gamma={self.g['Gamma']:.6f} is not positive"

    def test_vega_positive(self):
        """Vega must be positive (call price increases with volatility)."""
        assert self.g["Vega"] > 0.0, f"Vega={self.g['Vega']:.6f} is not positive"

    def test_rho_positive_call(self):
        """Rho of a call must be positive (call benefits from higher rates)."""
        assert self.g["Rho"] > 0.0, f"Rho={self.g['Rho']:.6f} is not positive"

    # -- Finite-difference checks (tolerance 1e-4)

    def test_delta_finite_difference(self):
        """Delta ~ (C(S+h) - C(S-h)) / (2h)."""
        h = 0.01
        fd_delta = (
            black_scholes_call(S0 + h, K, T, r, SIGMA, Q)
            - black_scholes_call(S0 - h, K, T, r, SIGMA, Q)
        ) / (2.0 * h)
        tol = 1e-4
        assert abs(self.g["Delta"] - fd_delta) < tol, (
            f"Delta FD error: analytic={self.g['Delta']:.6f}, FD={fd_delta:.6f}, "
            f"diff={abs(self.g['Delta']-fd_delta):.2e}"
        )

    def test_gamma_finite_difference(self):
        """Gamma ~ (C(S+h) - 2*C(S) + C(S-h)) / h^2."""
        h = 0.01
        c_up = black_scholes_call(S0 + h, K, T, r, SIGMA, Q)
        c_mid = black_scholes_call(S0, K, T, r, SIGMA, Q)
        c_dn = black_scholes_call(S0 - h, K, T, r, SIGMA, Q)
        fd_gamma = (c_up - 2.0 * c_mid + c_dn) / (h ** 2)
        tol = 1e-4
        assert abs(self.g["Gamma"] - fd_gamma) < tol, (
            f"Gamma FD error: analytic={self.g['Gamma']:.6f}, FD={fd_gamma:.6f}, "
            f"diff={abs(self.g['Gamma']-fd_gamma):.2e}"
        )

    def test_vega_finite_difference(self):
        """Vega ~ (C(sigma+h) - C(sigma-h)) / (2h)."""
        h = 0.001
        fd_vega = (
            black_scholes_call(S0, K, T, r, SIGMA + h, Q)
            - black_scholes_call(S0, K, T, r, SIGMA - h, Q)
        ) / (2.0 * h)
        tol = 1e-4
        assert abs(self.g["Vega"] - fd_vega) < tol, (
            f"Vega FD error: analytic={self.g['Vega']:.6f}, FD={fd_vega:.6f}, "
            f"diff={abs(self.g['Vega']-fd_vega):.2e}"
        )

    def test_rho_finite_difference(self):
        """Rho ~ (C(r+h) - C(r-h)) / (2h)."""
        h = 0.0001
        fd_rho = (
            black_scholes_call(S0, K, T, r + h, SIGMA, Q)
            - black_scholes_call(S0, K, T, r - h, SIGMA, Q)
        ) / (2.0 * h)
        tol = 1e-4
        assert abs(self.g["Rho"] - fd_rho) < tol, (
            f"Rho FD error: analytic={self.g['Rho']:.6f}, FD={fd_rho:.6f}, "
            f"diff={abs(self.g['Rho']-fd_rho):.2e}"
        )

    # -- Known-value sanity check

    def test_put_call_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T)."""
        call = black_scholes_call(S0, K, T, r, SIGMA, Q)
        put = black_scholes_put(S0, K, T, r, SIGMA, Q)
        parity = S0 * np.exp(-Q * T) - K * np.exp(-r * T)
        assert abs(call - put - parity) < 1e-10, (
            f"Put-call parity violated: C-P={call-put:.8f}, PCP={parity:.8f}"
        )

    def test_digital_between_zero_and_discount(self):
        """Digital call price in [0, exp(-r*T)]."""
        price = black_scholes_digital_call(S0, K, T, r, SIGMA, Q)
        assert 0.0 < price < np.exp(-r * T), f"Digital price {price:.6f} out of bounds"

    def test_lookback_analytic_known_value(self):
        """GSG formula: S=100,T=1,r=0.05,q=0,sigma=0.2 -> V ~ 14.29 (continuous-time)."""
        V = lookback_put_floating_analytic(100.0, 1.0, 0.05, 0.20, 0.0)
        # Verified by MC extrapolation to continuous limit (n_steps->inf converges to ~14.29).
        assert abs(V - 14.29) < 0.1, f"GSG known value: expected ~14.29, got {V:.4f}"

    def test_barrier_doc_less_than_vanilla(self):
        """DOC price must be <= vanilla call price (barrier reduces payoff)."""
        B = 85.0
        doc = barrier_down_and_out_call_analytic(S0, K, B, T, r, SIGMA, Q)
        vanilla = black_scholes_call(S0, K, T, r, SIGMA, Q)
        assert doc <= vanilla + 1e-10, (
            f"DOC={doc:.4f} > vanilla={vanilla:.4f}, which is impossible"
        )
        assert doc >= 0.0, f"DOC price {doc:.4f} is negative"

    def test_geometric_asian_less_than_vanilla(self):
        """Geometric Asian call <= vanilla call (lower average <= terminal value)."""
        asian = asian_geometric_call(S0, K, T, r, SIGMA, 252, Q)
        vanilla = black_scholes_call(S0, K, T, r, SIGMA, Q)
        assert asian <= vanilla + 1e-6, (
            f"Asian={asian:.4f} > vanilla={vanilla:.4f}"
        )
        assert asian >= 0.0, f"Asian price {asian:.4f} is negative"
