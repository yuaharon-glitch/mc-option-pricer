"""
payoffs.py — Vectorised discounted payoff functions.

All functions take paths (n_paths, n_steps+1) and return shape (n_paths,).
paths[:, 0] = S0. Asian averages exclude the initial price column.
"""

from __future__ import annotations

import numpy as np


def _validate_paths(paths, name="paths"):
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError(f"{name} must be 2-D (n_paths, n_steps+1), got {paths.shape}")


def _disc(r, T):
    return float(np.exp(-r * T))


def european_call(paths, K, r, T):
    _validate_paths(paths)
    return _disc(r, T) * np.maximum(paths[:, -1] - K, 0.0)


def european_put(paths, K, r, T):
    _validate_paths(paths)
    return _disc(r, T) * np.maximum(K - paths[:, -1], 0.0)


def asian_call_arithmetic(paths, K, r, T):
    """Arithmetic average Asian call. Average over monitoring dates t_1..t_n."""
    _validate_paths(paths)
    return _disc(r, T) * np.maximum(paths[:, 1:].mean(axis=1) - K, 0.0)


def asian_call_geometric(paths, K, r, T):
    """Geometric average Asian call."""
    _validate_paths(paths)
    G = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    return _disc(r, T) * np.maximum(G - K, 0.0)


def barrier_down_and_out_call(paths, K, B, r, T):
    """DOC: knocked out if min(path) <= B at any monitored step."""
    _validate_paths(paths)
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    payoff[paths.min(axis=1) <= B] = 0.0
    return _disc(r, T) * payoff


def barrier_up_and_out_call(paths, K, B, r, T):
    _validate_paths(paths)
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    payoff[paths.max(axis=1) >= B] = 0.0
    return _disc(r, T) * payoff


def lookback_call_fixed(paths, K, r, T):
    _validate_paths(paths)
    return _disc(r, T) * np.maximum(paths.max(axis=1) - K, 0.0)


def lookback_put_floating(paths, r, T):
    """Floating lookback put: max(S_t) - S_T."""
    _validate_paths(paths)
    return _disc(r, T) * np.maximum(paths.max(axis=1) - paths[:, -1], 0.0)


def digital_call(paths, K, r, T):
    """Cash-or-nothing digital: pays 1 if S_T > K."""
    _validate_paths(paths)
    return _disc(r, T) * (paths[:, -1] > K).astype(np.float64)
