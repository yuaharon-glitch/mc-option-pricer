"""mc_pricer — Monte Carlo option pricing with variance reduction."""

from __future__ import annotations

__version__ = "0.1.0"

# Global random seed — used by all modules via np.random.default_rng(RANDOM_SEED)
RANDOM_SEED: int = 42

__all__ = [
    "RANDOM_SEED",
    "__version__",
]
