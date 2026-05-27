"""
analytics.py — Closed-form option prices, Greeks, and implied volatility.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


def _d1_d2(S, K, T, r, sigma, q=0.0):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(d1), float(d1 - sigma * sqrt_T)


def _npdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)




def black_scholes_call(S0, K, T, r, sigma, q=0.0):
    if T <= 0.0:
        return float(max(S0 - K, 0.0))
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    return float(S0 * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2))


def black_scholes_put(S0, K, T, r, sigma, q=0.0):
    if T <= 0.0:
        return float(max(K - S0, 0.0))
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    return float(K * np.exp(-r * T) * ndtr(-d2) - S0 * np.exp(-q * T) * ndtr(-d1))


def black_scholes_digital_call(S0, K, T, r, sigma, q=0.0):
    """Cash-or-nothing digital: pays 1 if S_T > K."""
    if T <= 0.0:
        return float(1.0 if S0 > K else 0.0)
    _, d2 = _d1_d2(S0, K, T, r, sigma, q)
    return float(np.exp(-r * T) * ndtr(d2))


# asian geometric call (exact, discrete monitoring)

def asian_geometric_call(S0, K, T, r, sigma, n_steps, q=0.0):
    """Exact price for discretely-monitored geometric Asian call.

    The geometric average G_n = (∏ S_{t_k})^{1/n} is lognormal with:
        sigma_G² = sigma² * T * (2n+1) / (6n)
        mu_G = log(S0) + (r-q-sigma²/2) * T*(n+1)/(2n)

    Priced as a Black-Scholes call on the synthetic forward F_G = exp(mu_G + sigma_G²/2).
    """
    b = r - q
    n = n_steps
    sigma_G = sigma * np.sqrt(T * (2*n + 1) / (6*n))
    mu_G = np.log(S0) + (b - 0.5 * sigma**2) * T * (n + 1) / (2*n)
    F_G = np.exp(mu_G + 0.5 * sigma_G**2)

    if sigma_G < 1e-14:
        return float(np.exp(-r * T) * max(F_G - K, 0.0))

    d1 = (np.log(F_G / K) + 0.5 * sigma_G**2) / sigma_G
    d2 = d1 - sigma_G
    return float(np.exp(-r * T) * (F_G * ndtr(d1) - K * ndtr(d2)))


# lookback floating put (Goldman-Sosin-Gatto 1979)

def lookback_put_floating_analytic(S0, T, r, sigma, q=0.0):
    """Floating lookback put: V = e^{-rT} E[max S_t - S_T].

    Assumes M_0 = S_0 (pricing at inception). Requires r != q.

    Derived from the MGF of the GBM running maximum via the reflection
    principle with drift:

        E[M_T]/S0 = e^{bT}*(1+lam)*N(a1) + (1-lam)*N(-a2)
        V = S0*e^{-qT}*[(1+lam)*N(a1) - 1] + S0*e^{-rT}*(1-lam)*N(-a2)

    where b=r-q, a1=(b+sigma²/2)*sqrt(T)/sigma, a2=a1-sigma*sqrt(T),
    lam=sigma²/(2b).

    Verified: S0=100, r=0.05, q=0, sigma=0.2, T=1 → V ≈ 14.29.
    """
    b = r - q
    if abs(b) < 1e-12:
        raise ValueError("r must differ from q (b = r-q cannot be zero).")

    sqrt_T = np.sqrt(T)
    a1 = (b + 0.5 * sigma**2) * sqrt_T / sigma
    a2 = a1 - sigma * sqrt_T
    lam = sigma**2 / (2.0 * b)

    V = (
        S0 * np.exp(-q * T) * (1.0 + lam) * ndtr(a1)
        - S0 * np.exp(-q * T)
        + S0 * np.exp(-r * T) * (1.0 - lam) * ndtr(-a2)
    )
    return float(V)


# Barrier: down-and-out call (Merton 1973)

def barrier_down_and_out_call_analytic(S0, K, B, T, r, sigma, q=0.0):
    """Merton (1973) DOC price via reflection principle (continuous monitoring).

    V = C_vanilla - (B/S)^(2*mu) * C_image
    where mu = (r-q-sigma²/2)/sigma², and C_image is a vanilla call
    on the reflected asset B²/S.

    Requires S0 > B.
    """
    if S0 <= B:
        raise ValueError(f"S0={S0} must be > barrier B={B}.")

    b = r - q
    mu = (b - 0.5 * sigma**2) / sigma**2
    sqrt_T = np.sqrt(T)

    x1 = np.log(S0 / K) / (sigma * sqrt_T) + (1.0 + mu) * sigma * sqrt_T
    y1 = np.log(B**2 / (S0 * K)) / (sigma * sqrt_T) + (1.0 + mu) * sigma * sqrt_T

    v1 = S0 * np.exp(-q * T) * ndtr(x1)
    v2 = K * np.exp(-r * T) * ndtr(x1 - sigma * sqrt_T)
    v3 = S0 * np.exp(-q * T) * (B / S0) ** (2.0 * (mu + 1.0)) * ndtr(y1)
    v4 = K * np.exp(-r * T) * (B / S0) ** (2.0 * mu) * ndtr(y1 - sigma * sqrt_T)

    return float(v1 - v2 - v3 + v4)




def implied_volatility(market_price, S0, K, T, r, option_type="call", q=0.0,
                       tol=1e-10, maxiter=500):
    """Implied vol via Brent's method. Raises ValueError if bracketing fails."""
    if option_type == "call":
        f = lambda s: black_scholes_call(S0, K, T, r, s, q) - market_price
    elif option_type == "put":
        f = lambda s: black_scholes_put(S0, K, T, r, s, q) - market_price
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    lo, hi = 1e-6, 10.0
    if f(lo) * f(hi) > 0.0:
        raise ValueError(
            f"IV bracket failed: f({lo})={f(lo):.4f}, f({hi})={f(hi):.4f}. "
            f"market_price={market_price:.4f} may be outside attainable range."
        )
    return float(brentq(f, lo, hi, xtol=tol, maxiter=maxiter))


# Greeks

def greeks(S0, K, T, r, sigma, q=0.0):
    """Analytical Black-Scholes Greeks for a European call.

    Returns dict with Delta, Gamma, Vega, Theta, Rho.
    """
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    nd1 = _npdf(d1)
    eq = np.exp(-q * T)
    er = np.exp(-r * T)

    return {
        "Delta": float(eq * ndtr(d1)),
        "Gamma": float(eq * nd1 / (S0 * sigma * sqrt_T)),
        "Vega":  float(S0 * eq * nd1 * sqrt_T),
        "Theta": float(
            -S0 * eq * nd1 * sigma / (2.0 * sqrt_T)
            + q * S0 * eq * ndtr(d1)
            - r * K * er * ndtr(d2)
        ),
        "Rho":   float(K * T * er * ndtr(d2)),
    }
