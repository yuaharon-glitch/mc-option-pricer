"""
processes.py — Stochastic process simulation (GBM exact scheme, Heston Milstein).
"""

from __future__ import annotations

import warnings

import numpy as np

from mc_pricer import RANDOM_SEED


class GeometricBrownianMotion:
    """Exact log-Euler simulation of GBM under the risk-neutral measure.

    Parameters
    ----------
    S0 : float
    mu : float
        Physical drift (stored but not used in simulation — r-q is used).
    sigma : float
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    """

    def __init__(self, S0, mu, sigma, r, q=0.0):
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.r = float(r)
        self.q = float(q)

    def simulate(self, T, n_steps, n_paths, rng=None):
        """Simulate paths using the exact log-Euler scheme.

        Returns
        -------
        np.ndarray, shape (n_paths, n_steps + 1)
        """
        rng = rng or np.random.default_rng(RANDOM_SEED)
        Z = rng.standard_normal((n_paths, n_steps))
        return self._simulate_from_normals(Z, T, n_steps)

    def simulate_antithetic(self, T, n_steps, n_paths, rng=None):
        """Simulate antithetic path pairs.

        Generates n_paths//2 standard paths and their -Z mirrors.
        Standard paths occupy rows [:half], antithetic rows [half:].

        Returns
        -------
        np.ndarray, shape (n_paths, n_steps + 1)
        """
        rng = rng or np.random.default_rng(RANDOM_SEED)
        if n_paths % 2:
            n_paths += 1
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        std = self._simulate_from_normals(Z, T, n_steps)
        anti = self._simulate_from_normals(-Z, T, n_steps)
        result = np.empty((n_paths, n_steps + 1))
        result[:half] = std
        result[half:] = anti
        return result

    def terminal_distribution(self, T, n_paths, rng=None):
        """Sample terminal prices S(T) directly (no intermediate steps).

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        rng = rng or np.random.default_rng(RANDOM_SEED)
        Z = rng.standard_normal(n_paths)
        return self.S0 * np.exp(
            (self.r - self.q - 0.5 * self.sigma**2) * T
            + self.sigma * np.sqrt(T) * Z
        )

    def _simulate_from_normals(self, Z, T, n_steps):
        """Build paths from pre-supplied standard normals Z (n_paths, n_steps)."""
        dt = T / n_steps
        log_inc = (self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        log_S = np.log(self.S0) + np.cumsum(log_inc, axis=1)
        S0_col = np.full((Z.shape[0], 1), self.S0)
        return np.concatenate([S0_col, np.exp(log_S)], axis=1)


class HestonProcess:
    """Heston (1993) stochastic-vol model simulated via Milstein with full truncation.

    SDEs (risk-neutral):
        dS = r*S dt + sqrt(v)*S dW_S
        dv = kappa*(theta - v) dt + xi*sqrt(v) dW_v,   Corr(dW_S, dW_v) = rho

    Full truncation: v_plus = max(v, 0) in drift/diffusion; v itself may
    go transiently negative. Milstein correction term: 0.25*xi²*dt*(Z_v²-1).

    Parameters
    ----------
    S0, v0, kappa, theta, xi, rho, r : float
    """

    def __init__(self, S0, v0, kappa, theta, xi, rho, r):
        self.S0 = float(S0)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.r = float(r)

        if 2.0 * kappa * theta < xi**2:
            warnings.warn(
                f"Feller condition violated: 2*kappa*theta={2*kappa*theta:.4f} < xi^2={xi**2:.4f}. "
                "Variance may hit zero; full truncation applied.",
                UserWarning,
                stacklevel=2,
            )

    def simulate(self, T, n_steps, n_paths, scheme="milstein", rng=None):
        """Simulate correlated (S, v) paths.

        Returns
        -------
        S_paths : np.ndarray, shape (n_paths, n_steps + 1)
        v_paths : np.ndarray, shape (n_paths, n_steps + 1)
        """
        rng = rng or np.random.default_rng(RANDOM_SEED)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        rho2_comp = np.sqrt(1.0 - self.rho**2)

        S = np.full(n_paths, self.S0)
        v = np.full(n_paths, self.v0)

        S_paths = np.empty((n_paths, n_steps + 1))
        v_paths = np.empty((n_paths, n_steps + 1))
        S_paths[:, 0] = S
        v_paths[:, 0] = v

        use_milstein = (scheme == "milstein")
        Z_all = rng.standard_normal((2, n_paths, n_steps))

        for i in range(n_steps):
            Z1, Z2 = Z_all[0, :, i], Z_all[1, :, i]
            Z_S = Z1
            Z_v = self.rho * Z1 + rho2_comp * Z2

            v_plus = np.maximum(v, 0.0)
            sv_dt = np.sqrt(v_plus * dt)

            correction = 0.25 * self.xi**2 * dt * (Z_v**2 - 1.0) if use_milstein else 0.0
            v = v + self.kappa * (self.theta - v_plus) * dt + self.xi * sv_dt * Z_v + correction
            S = S * np.exp((self.r - 0.5 * v_plus) * dt + sv_dt * Z_S)

            S_paths[:, i + 1] = S
            v_paths[:, i + 1] = v

        return S_paths, v_paths
