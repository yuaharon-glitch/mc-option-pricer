"""Tests for payoffs.py — shape, sign, and boundary checks."""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricer.payoffs import (
    asian_call_arithmetic,
    asian_call_geometric,
    barrier_down_and_out_call,
    barrier_up_and_out_call,
    digital_call,
    european_call,
    european_put,
    lookback_call_fixed,
    lookback_put_floating,
)

# Toy paths: 4 paths × 5 time steps (including t=0)
PATHS = np.array([
    [100.0, 102.0, 104.0, 103.0, 106.0],   # ends ITM for call K=100
    [100.0,  98.0,  96.0,  94.0,  92.0],   # ends OTM for call K=100
    [100.0, 105.0,  85.0, 110.0, 115.0],   # goes below 90 -> DOC knocked out
    [100.0,  99.0, 101.0, 100.0, 100.0],   # ends ATM
], dtype=float)

R = 0.05
T = 1.0
K = 100.0
DISC = np.exp(-R * T)


class TestEuropean:
    def test_call_shape(self):
        p = european_call(PATHS, K, R, T)
        assert p.shape == (4,)

    def test_call_nonneg(self):
        p = european_call(PATHS, K, R, T)
        assert (p >= 0).all()

    def test_call_values(self):
        p = european_call(PATHS, K, R, T)
        expected = DISC * np.maximum(PATHS[:, -1] - K, 0.0)
        np.testing.assert_allclose(p, expected)

    def test_put_values(self):
        p = european_put(PATHS, K, R, T)
        expected = DISC * np.maximum(K - PATHS[:, -1], 0.0)
        np.testing.assert_allclose(p, expected)

    def test_put_call_parity_payoffs(self):
        """call - put = disc * (S_T - K)."""
        call = european_call(PATHS, K, R, T)
        put  = european_put(PATHS, K, R, T)
        diff = call - put
        expected = DISC * (PATHS[:, -1] - K)
        np.testing.assert_allclose(diff, expected, atol=1e-12)


class TestAsian:
    def test_arithmetic_shape(self):
        p = asian_call_arithmetic(PATHS, K, R, T)
        assert p.shape == (4,)

    def test_arithmetic_nonneg(self):
        p = asian_call_arithmetic(PATHS, K, R, T)
        assert (p >= 0).all()

    def test_arithmetic_excludes_t0(self):
        # Path 1 avg = (102+104+103+106)/4 = 103.75 > 100
        p = asian_call_arithmetic(PATHS, K, R, T)
        expected_0 = DISC * max(PATHS[0, 1:].mean() - K, 0.0)
        np.testing.assert_allclose(p[0], expected_0, rtol=1e-10)

    def test_geometric_le_arithmetic(self):
        """Geometric mean <= arithmetic mean (AM-GM inequality)."""
        p_arith = asian_call_arithmetic(PATHS, K, R, T)
        p_geo   = asian_call_geometric(PATHS, K, R, T)
        assert (p_geo <= p_arith + 1e-12).all()

    def test_geometric_nonneg(self):
        p = asian_call_geometric(PATHS, K, R, T)
        assert (p >= 0).all()


class TestBarrier:
    def test_doc_knocked_out_path(self):
        """Path 3 (min=85 < B=90) should give zero payoff."""
        B = 90.0
        p = barrier_down_and_out_call(PATHS, K, B, R, T)
        assert p[2] == 0.0, f"Path 3 should be knocked out, got {p[2]}"

    def test_doc_surviving_path(self):
        """Path 0 never hits barrier B=90, ends ITM -> positive payoff."""
        B = 90.0
        p = barrier_down_and_out_call(PATHS, K, B, R, T)
        assert p[0] > 0.0

    def test_doc_le_vanilla(self):
        """DOC payoff <= vanilla call payoff for each path."""
        B = 90.0
        doc = barrier_down_and_out_call(PATHS, K, B, R, T)
        vanilla = european_call(PATHS, K, R, T)
        assert (doc <= vanilla + 1e-12).all()

    def test_uoc_knocked_out(self):
        """Path 4 (max=115 >= B=110) should give zero payoff."""
        B = 110.0
        p = barrier_up_and_out_call(PATHS, K, B, R, T)
        assert p[3] == 0.0 or p[2] == 0.0  # path 2 reaches 110

    def test_uoc_nonneg(self):
        p = barrier_up_and_out_call(PATHS, K, 200.0, R, T)
        assert (p >= 0).all()


class TestLookback:
    def test_fixed_call_shape(self):
        p = lookback_call_fixed(PATHS, K, R, T)
        assert p.shape == (4,)

    def test_fixed_call_nonneg(self):
        p = lookback_call_fixed(PATHS, K, R, T)
        assert (p >= 0).all()

    def test_fixed_call_values(self):
        """max(max_t S - K, 0) for each path."""
        p = lookback_call_fixed(PATHS, K, R, T)
        expected = DISC * np.maximum(PATHS.max(axis=1) - K, 0.0)
        np.testing.assert_allclose(p, expected)

    def test_floating_put_nonneg(self):
        p = lookback_put_floating(PATHS, R, T)
        assert (p >= 0).all()

    def test_floating_put_values(self):
        """max_t S - S_T for each path (always >= 0)."""
        p = lookback_put_floating(PATHS, R, T)
        expected = DISC * (PATHS.max(axis=1) - PATHS[:, -1])
        np.testing.assert_allclose(p, expected, atol=1e-12)


class TestDigital:
    def test_shape(self):
        p = digital_call(PATHS, K, R, T)
        assert p.shape == (4,)

    def test_binary_values(self):
        """Payoff is either 0 or exp(-r*T)."""
        p = digital_call(PATHS, K, R, T)
        unique = np.unique(p)
        for v in unique:
            assert abs(v) < 1e-12 or abs(v - DISC) < 1e-12

    def test_itm_path_pays_disc(self):
        """Path 0 ends at 106 > 100, should get full discount."""
        p = digital_call(PATHS, K, R, T)
        np.testing.assert_allclose(p[0], DISC)

    def test_otm_path_pays_zero(self):
        """Path 1 ends at 92 < 100, should get zero."""
        p = digital_call(PATHS, K, R, T)
        assert p[1] == 0.0


class TestValidation:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            european_call(np.array([100.0, 102.0]), K, R, T)

    def test_single_column_raises(self):
        with pytest.raises(ValueError):
            european_call(np.array([[100.0]]), K, R, T)
