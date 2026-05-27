"""
variance_reduction.py — Variance reduction utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from mc_pricer import RANDOM_SEED


def _cv_beta_star(f_samples, g_samples):
    """Estimate b* = Cov(f,g)/Var(g) and correlation rho.

    Returns
    -------
    b_star : float
    rho : float
    """
    cov = np.cov(f_samples, g_samples, ddof=1)
    var_g = cov[1, 1]
    if var_g < 1e-20:
        return 0.0, 0.0
    b_star = cov[0, 1] / var_g
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * var_g + 1e-30)
    return float(b_star), float(rho)


def stratified_sampling(n_strata, n_per_stratum, rng=None):
    """Stratified uniform sampling → standard normals.

    Divides [0,1) into n_strata equal bins. Draws n_per_stratum uniforms
    per bin and maps to normals.

    Returns
    -------
    np.ndarray, shape (n_strata, n_per_stratum)
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    k = np.arange(n_strata, dtype=np.float64)
    u_local = rng.uniform(0.0, 1.0, size=(n_strata, n_per_stratum))
    u = (k[:, np.newaxis] + u_local) / n_strata
    return ndtri(np.clip(u, 1e-10, 1.0 - 1e-10))


def latin_hypercube_sampling(n_samples, n_dims, rng=None):
    """Latin Hypercube Sampling → standard normals.

    Exactly one sample per stratum per dimension, with random within-stratum
    offsets and independent permutations per dimension.

    Returns
    -------
    np.ndarray, shape (n_samples, n_dims)
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    u = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        offset = rng.uniform(0.0, 1.0, n_samples)
        u[:, d] = (perm + offset) / n_samples
    return ndtri(np.clip(u, 1e-10, 1.0 - 1e-10))


def compute_vrr_table(results, baseline_method="plain_mc"):
    """Build a VRR summary DataFrame from a dict of MCResult objects.

    Parameters
    ----------
    results : dict[str, MCResult]
    baseline_method : str

    Returns
    -------
    pd.DataFrame
        Columns: method, price, std_error, variance, VRR.
    """
    if baseline_method not in results:
        raise KeyError(
            f"baseline_method='{baseline_method}' not in results. "
            f"Available: {list(results.keys())}"
        )
    base_var = results[baseline_method].variance
    rows = []
    for method, res in results.items():
        vrr = base_var / res.variance if res.variance > 0 else float("nan")
        rows.append({
            "method": method,
            "price": res.price,
            "std_error": res.std_error,
            "variance": res.variance,
            "VRR": vrr,
        })
    return pd.DataFrame(rows).set_index("method")
