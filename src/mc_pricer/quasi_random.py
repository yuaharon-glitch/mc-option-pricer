"""
quasi_random.py — Sobol quasi-random sequence generator (from scratch).

Uses bit arithmetic and direction numbers from Joe & Kuo (2010).
No scipy.stats.qmc used for sequence generation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


# Joe & Kuo (2010) direction number init data, dims 2–21.
_JOE_KUO_DATA = [
    (1, 0, [1]),
    (2, 1, [1, 1]),
    (3, 1, [1, 1, 1]),
    (3, 2, [1, 3, 7]),
    (4, 1, [1, 1, 5, 3]),
    (4, 4, [1, 3, 1, 1]),
    (5, 2, [1, 1, 3, 5, 5]),
    (5, 4, [1, 1, 5, 5, 7]),
    (5, 7, [1, 1, 5, 7, 7]),
    (5, 11, [1, 1, 7, 3, 3]),
    (5, 13, [1, 1, 7, 5, 1]),
    (5, 14, [1, 3, 5, 3, 5]),
    (6, 1, [1, 3, 5, 7, 1, 3]),
    (6, 13, [1, 1, 1, 1, 7, 9]),
    (6, 16, [1, 1, 3, 7, 3, 11]),
    (6, 19, [1, 1, 3, 7, 9, 5]),
    (6, 22, [1, 3, 5, 7, 7, 11]),
    (6, 25, [1, 3, 7, 1, 5, 11]),
    (7, 1, [1, 1, 1, 3, 1, 7, 3]),
    (7, 4, [1, 1, 1, 3, 3, 9, 7]),
]

_W = 32
_SCALE = 2.0 ** (-_W)


def _build_direction_numbers(dimension):
    if dimension < 1 or dimension > 21:
        raise ValueError(f"dimension must be in [1, 21], got {dimension}")

    V = np.zeros((_W, dimension), dtype=np.uint32)

    for j in range(1, _W + 1):
        V[j - 1, 0] = np.uint32(1 << (_W - j))

    for dim in range(1, dimension):
        s, a, m_init = _JOE_KUO_DATA[dim - 1]
        for j in range(1, s + 1):
            V[j - 1, dim] = np.uint32(int(m_init[j - 1]) << (_W - j))
        for j in range(s + 1, _W + 1):
            v_j = V[j - s - 1, dim] ^ (V[j - s - 1, dim] >> np.uint32(s))
            for k in range(1, s):
                if (a >> (s - 1 - k)) & 1:
                    v_j ^= V[j - k - 1, dim]
            V[j - 1, dim] = v_j

    return V


class SobolEngine:
    """Sobol low-discrepancy sequence generator using Joe & Kuo (2010) direction numbers.

    Parameters
    ----------
    dimension : int
        1 to 21.
    scramble : bool
        XOR scramble each dimension with a random uint32 mask.
    seed : int or None
    """

    def __init__(self, dimension, scramble=True, seed=None):
        from mc_pricer import RANDOM_SEED as _DEFAULT_SEED

        if dimension < 1 or dimension > 21:
            raise ValueError(f"dimension must be in [1, 21], got {dimension}")

        self.dimension = dimension
        self.scramble = scramble
        self._seed = seed if seed is not None else _DEFAULT_SEED
        self._V = _build_direction_numbers(dimension)

        if scramble:
            _rng = np.random.default_rng(self._seed)
            self._masks = _rng.integers(0, 2**_W, size=dimension, dtype=np.uint64).astype(np.uint32)
        else:
            self._masks = np.zeros(dimension, dtype=np.uint32)

        self.reset()

    def reset(self):
        self._x = np.zeros(self.dimension, dtype=np.uint32)
        self._count = 0

    def random(self, n):
        """Generate next n Sobol points.

        Returns
        -------
        np.ndarray, shape (n, dimension), float64 in [0, 1)
        """
        out = np.empty((n, self.dimension), dtype=np.uint64)
        for i in range(n):
            if self._count == 0:
                out[i] = 0
            else:
                c = min(int(self._count & -self._count).bit_length() - 1, _W - 1)
                self._x ^= self._V[c]
                out[i] = self._x
            self._count += 1
        out ^= self._masks[np.newaxis, :].astype(np.uint64)
        return out.astype(np.float64) * _SCALE

    def __call__(self, n):
        return self.random(n)


def sobol_to_normal(u, eps=1e-10):
    """Transform Sobol uniform samples to standard normals via inverse CDF."""
    return ndtri(np.clip(u, eps, 1.0 - eps))


def discrepancy(u):
    """1D star discrepancy D*_N of a point set in [0, 1)."""
    u_s = np.sort(np.ravel(u))
    N = len(u_s)
    k = np.arange(1, N + 1, dtype=np.float64)
    return float(np.maximum(k / N - u_s, u_s - (k - 1.0) / N).max())
