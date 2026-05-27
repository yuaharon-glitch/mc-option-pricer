"""
engines.py — Monte Carlo pricing engines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mc_pricer import RANDOM_SEED
from mc_pricer.processes import GeometricBrownianMotion
from mc_pricer.quasi_random import SobolEngine, sobol_to_normal
from mc_pricer.variance_reduction import _cv_beta_star


@dataclass
class MCResult:
    """Container for a Monte Carlo pricing estimate.

    Attributes
    ----------
    price : float
    std_error : float
    confidence_interval : tuple[float, float]
        95% CI.
    n_paths : int
    variance : float
        Sample variance (ddof=1).
    method : str
    n_paths_effective : int
        Effective sample size used for SE.
    """

    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    n_paths: int
    variance: float
    method: str
    n_paths_effective: int

    def variance_reduction_ratio(self, baseline: "MCResult") -> float:
        """VRR = Var(baseline) / Var(self)."""
        if self.variance <= 0.0:
            return float("nan")
        return baseline.variance / self.variance

    def efficiency_ratio(self, baseline: "MCResult", cost_ratio: float = 1.0) -> float:
        return self.variance_reduction_ratio(baseline) / cost_ratio


class MonteCarloEngine:
    """GBM-based MC pricing engine with five variance reduction methods.

    Parameters
    ----------
    process : GeometricBrownianMotion
    seed : int
    """

    def __init__(self, process: GeometricBrownianMotion, seed: int = RANDOM_SEED):
        self.process = process
        self.seed = seed

    def _call_payoff(self, payoff_fn, paths, T, **kw):
        kw = dict(kw)
        kw.setdefault("T", T)
        return payoff_fn(paths, **kw)

    def _build_result(self, payoffs, method, n_requested):
        n = len(payoffs)
        mean = float(payoffs.mean())
        var = float(payoffs.var(ddof=1))
        se = float(np.sqrt(var / n))
        return MCResult(
            price=mean,
            std_error=se,
            confidence_interval=(mean - 1.96 * se, mean + 1.96 * se),
            n_paths=n_requested,
            variance=var,
            method=method,
            n_paths_effective=n,
        )



    def price(self, payoff_fn, T, n_steps, n_paths, **kw):
        rng = np.random.default_rng(self.seed)
        paths = self.process.simulate(T, n_steps, n_paths, rng=rng)
        return self._build_result(self._call_payoff(payoff_fn, paths, T, **kw), "plain_mc", n_paths)



    def price_antithetic(self, payoff_fn, T, n_steps, n_paths, **kw):
        rng = np.random.default_rng(self.seed)
        if n_paths % 2:
            n_paths += 1
        half = n_paths // 2
        paths = self.process.simulate_antithetic(T, n_steps, n_paths, rng=rng)
        all_payoffs = self._call_payoff(payoff_fn, paths, T, **kw)
        paired = 0.5 * (all_payoffs[:half] + all_payoffs[half:])
        return self._build_result(paired, "antithetic", n_paths)

    # control variates

    def price_control_variate(
        self,
        payoff_fn,
        control_payoff_fn,
        control_analytic_price,
        T,
        n_steps,
        n_paths,
        pilot_paths=10_000,
        **kw,
    ):
        """Control variate: f_cv = f - b*(g - E[g]).

        b* is estimated from a pilot run, then applied to the full simulation.
        """
        rng_pilot = np.random.default_rng(self.seed)
        paths_p = self.process.simulate(T, n_steps, pilot_paths, rng=rng_pilot)
        f_p = self._call_payoff(payoff_fn, paths_p, T, **kw)
        g_p = self._call_payoff(control_payoff_fn, paths_p, T, **kw)
        b_star, rho = _cv_beta_star(f_p, g_p)

        rng_main = np.random.default_rng(self.seed + 1)
        paths = self.process.simulate(T, n_steps, n_paths, rng=rng_main)
        f = self._call_payoff(payoff_fn, paths, T, **kw)
        g = self._call_payoff(control_payoff_fn, paths, T, **kw)
        f_cv = f - b_star * (g - control_analytic_price)

        result = self._build_result(f_cv, "control_variate", n_paths)
        result.__dict__["_b_star"] = b_star
        result.__dict__["_rho"] = rho
        return result



    def price_importance_sampling(self, payoff_fn, T, n_steps, n_paths, theta, **kw):
        """Girsanov drift shift by theta.

        Optimal theta for a digital/call at strike K:
            theta* = (log(K/S0) - (r-q-sigma²/2)*T) / (sigma*sqrt(T))

        Likelihood ratio: log L = -theta*sqrt(dt)*sum(Z) - 0.5*theta²*T
        """
        rng = np.random.default_rng(self.seed)
        dt = T / n_steps
        Z = rng.standard_normal((n_paths, n_steps))
        paths = self.process._simulate_from_normals(Z + theta * np.sqrt(dt), T, n_steps)
        log_L = -theta * np.sqrt(dt) * Z.sum(axis=1) - 0.5 * theta**2 * T
        payoffs = self._call_payoff(payoff_fn, paths, T, **kw) * np.exp(log_L)
        return self._build_result(payoffs, "importance_sampling", n_paths)

    # quasi-MC (Sobol)

    def price_quasi_mc(self, payoff_fn, T, n_steps, n_paths, **kw):
        """Sobol QMC. Dimension capped at 21; remaining dims use pseudo-random normals."""
        dim = min(n_steps, 21)
        eng = SobolEngine(dimension=dim, scramble=True, seed=self.seed)
        Z_sobol = sobol_to_normal(eng.random(n_paths))

        if n_steps > dim:
            rng = np.random.default_rng(self.seed)
            Z = np.concatenate([Z_sobol, rng.standard_normal((n_paths, n_steps - dim))], axis=1)
        else:
            Z = Z_sobol

        paths = self.process._simulate_from_normals(Z, T, n_steps)
        payoffs = self._call_payoff(payoff_fn, paths, T, **kw)
        return self._build_result(payoffs, "quasi_mc", n_paths)



    def benchmark_all(
        self,
        payoff_fn,
        T,
        n_steps,
        n_paths_list,
        control_payoff_fn=None,
        control_analytic_price=None,
        is_theta=0.0,
        **kw,
    ):
        """Run all five methods across multiple path counts.

        Returns pd.DataFrame with columns:
            n_paths, method, price, std_error, variance, vrr, runtime_s
        """
        rows = []
        kw = {k: v for k, v in kw.items() if k != "T"}

        for n_paths in n_paths_list:
            results = {}

            t0 = time.perf_counter()
            results["plain_mc"] = self.price(payoff_fn, T, n_steps, n_paths, **kw)
            t_plain = time.perf_counter() - t0

            t0 = time.perf_counter()
            results["antithetic"] = self.price_antithetic(payoff_fn, T, n_steps, n_paths, **kw)
            t_anti = time.perf_counter() - t0

            t_cv = None
            if control_payoff_fn is not None and control_analytic_price is not None:
                t0 = time.perf_counter()
                results["control_variate"] = self.price_control_variate(
                    payoff_fn, control_payoff_fn, control_analytic_price,
                    T, n_steps, n_paths, **kw
                )
                t_cv = time.perf_counter() - t0

            t0 = time.perf_counter()
            results["importance_sampling"] = self.price_importance_sampling(
                payoff_fn, T, n_steps, n_paths, theta=is_theta, **kw
            )
            t_is = time.perf_counter() - t0

            t0 = time.perf_counter()
            results["quasi_mc"] = self.price_quasi_mc(payoff_fn, T, n_steps, n_paths, **kw)
            t_qmc = time.perf_counter() - t0

            runtimes = {
                "plain_mc": t_plain, "antithetic": t_anti,
                "importance_sampling": t_is, "quasi_mc": t_qmc,
            }
            if t_cv is not None:
                runtimes["control_variate"] = t_cv

            base_var = results["plain_mc"].variance
            for method, res in results.items():
                vrr = base_var / res.variance if res.variance > 0 else float("nan")
                rows.append({
                    "n_paths": n_paths, "method": method,
                    "price": res.price, "std_error": res.std_error,
                    "variance": res.variance, "vrr": vrr,
                    "runtime_s": runtimes.get(method, float("nan")),
                })

        return pd.DataFrame(rows)
