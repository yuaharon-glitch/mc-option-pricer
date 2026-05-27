"""
benchmarks.py — Convergence and variance-reduction benchmarking utilities.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mc_pricer import RANDOM_SEED
from mc_pricer.engines import MCResult, MonteCarloEngine
from mc_pricer.processes import GeometricBrownianMotion


class ConvergenceBenchmark:
    """Run repeated MC estimates across multiple sample sizes.

    Parameters
    ----------
    engine : MonteCarloEngine
    payoff_fn : callable
    T : float
    n_steps : int
    analytic_price : float or None
    method : str
        One of: plain, antithetic, control_variate, importance_sampling, quasi_mc.
    method_kwargs : dict
        Extra args for control_variate (control_fn, control_price) or IS (theta).
    payoff_kwargs : dict
    """

    def __init__(
        self,
        engine,
        payoff_fn,
        T,
        n_steps,
        analytic_price=None,
        method="plain",
        method_kwargs=None,
        payoff_kwargs=None,
    ):
        self.engine = engine
        self.payoff_fn = payoff_fn
        self.T = T
        self.n_steps = n_steps
        self.analytic_price = analytic_price
        self.method = method
        self.method_kwargs = method_kwargs or {}
        self.payoff_kwargs = payoff_kwargs or {}

    def run(self, n_paths_list, n_repetitions=20, seed_offset=0):
        """Run repeated estimates for each sample size.

        Returns
        -------
        pd.DataFrame
            Long-format with columns: n_paths, repetition, price, std_error,
            variance, and rmse (if analytic_price is given).
        """
        rows = []
        orig_seed = self.engine.seed

        for n in n_paths_list:
            for rep in range(n_repetitions):
                self.engine.seed = RANDOM_SEED + seed_offset + rep * 1000
                try:
                    result = self._price_one(n)
                finally:
                    self.engine.seed = orig_seed

                row = {
                    "n_paths": n, "repetition": rep,
                    "price": result.price,
                    "std_error": result.std_error,
                    "variance": result.variance,
                }
                if self.analytic_price is not None:
                    row["rmse_contribution"] = (result.price - self.analytic_price) ** 2
                rows.append(row)

        df = pd.DataFrame(rows)
        if self.analytic_price is not None:
            rmse = (
                df.groupby("n_paths")["rmse_contribution"]
                .mean().apply(np.sqrt).rename("rmse").reset_index()
            )
            df = df.merge(rmse, on="n_paths", how="left")
        return df

    def _price_one(self, n_paths):
        m, kw = self.method, self.payoff_kwargs
        if m == "plain":
            return self.engine.price(self.payoff_fn, self.T, self.n_steps, n_paths, **kw)
        elif m == "antithetic":
            return self.engine.price_antithetic(self.payoff_fn, self.T, self.n_steps, n_paths, **kw)
        elif m == "control_variate":
            return self.engine.price_control_variate(
                self.payoff_fn, self.method_kwargs["control_fn"],
                self.method_kwargs["control_price"],
                self.T, self.n_steps, n_paths, **kw
            )
        elif m == "importance_sampling":
            return self.engine.price_importance_sampling(
                self.payoff_fn, self.T, self.n_steps, n_paths,
                theta=self.method_kwargs["theta"], **kw
            )
        elif m == "quasi_mc":
            return self.engine.price_quasi_mc(self.payoff_fn, self.T, self.n_steps, n_paths, **kw)
        else:
            raise ValueError(f"Unknown method '{m}'.")


def fit_convergence_rate(df, n_col="n_paths", rmse_col="rmse"):
    """Fit RMSE ~ c * N^beta via log-log OLS.

    Returns dict with beta, log_c, r2.
    The reported beta is empirical over the measured N range.
    """
    sub = df.dropna(subset=[n_col, rmse_col])
    sub = sub[sub[rmse_col] > 0]
    if len(sub) < 2:
        raise ValueError(f"Need >= 2 non-zero RMSE values, got {len(sub)}.")

    log_n = np.log(sub[n_col].values.astype(float))
    log_r = np.log(sub[rmse_col].values.astype(float))
    beta, log_c = np.polyfit(log_n, log_r, 1)

    pred = beta * log_n + log_c
    ss_res = np.sum((log_r - pred) ** 2)
    ss_tot = np.sum((log_r - log_r.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"beta": float(beta), "log_c": float(log_c), "r2": r2}


def plot_convergence(results_dict, title="MC Convergence", figsize=(9, 6), save_path=None):
    """Log-log RMSE vs N for multiple methods.

    Parameters
    ----------
    results_dict : dict[str, pd.DataFrame]
        Each DataFrame must have 'n_paths' and 'rmse' (aggregated per n_paths).
    save_path : str or None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, (name, df) in enumerate(results_dict.items()):
        if "rmse" not in df.columns:
            continue
        sub = df.groupby("n_paths")["rmse"].first().reset_index()
        sub = sub[sub["rmse"] > 0]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.loglog(sub["n_paths"], sub["rmse"], marker=m, color=c, label=name)

        try:
            fit = fit_convergence_rate(sub)
            n_vals = sub["n_paths"].values
            n_mid = np.sqrt(n_vals[0] * n_vals[-1])
            rmse_mid = np.exp(fit["log_c"]) * n_mid ** fit["beta"]
            n_ref = np.array([n_vals[0], n_vals[-1]])
            ax.loglog(
                n_ref, rmse_mid * (n_ref / n_mid) ** fit["beta"],
                "--", color=c, alpha=0.5,
                label=f"{name} β={fit['beta']:.2f}",
            )
        except Exception:
            pass

    # Reference lines
    n_all = np.concatenate([df["n_paths"].values for df in results_dict.values() if len(df) > 0])
    if len(n_all) > 0:
        n_ref = np.logspace(np.log10(n_all.min()), np.log10(n_all.max()), 100)
        first = next(iter(results_dict.values()))
        if "rmse" in first.columns:
            mid_row = first.groupby("n_paths")["rmse"].first().reset_index()
            mid_row = mid_row[mid_row["rmse"] > 0]
            if len(mid_row):
                anchor_n = np.sqrt(n_all.min() * n_all.max())
                anchor_r = mid_row["rmse"].iloc[len(mid_row) // 2]
                ax.loglog(n_ref, anchor_r * (n_ref / anchor_n) ** -0.5, ":", color="grey", alpha=0.7,
                          label=r"$N^{-1/2}$")
                ax.loglog(n_ref, anchor_r * (n_ref / anchor_n) ** -1.0, "-.", color="grey", alpha=0.7,
                          label=r"$N^{-1}$")

    ax.set_xlabel("N (paths)")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
