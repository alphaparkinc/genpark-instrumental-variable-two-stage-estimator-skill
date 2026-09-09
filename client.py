"""
Instrumental Variable Two-Stage Least Squares (2SLS) Estimator Skill Client
Pure Python Standard Library implementation of 2SLS causal estimation.
Uses an exogenous instrument Z to isolate exogenous variation in endogenous treatment X,
producing consistent and unbiased estimates of the causal effect of X on Y even under severe unobserved confounding.
"""

from typing import List, Tuple, Dict


def simple_linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Fit simple OLS y = beta0 + beta1 * x. Returns (beta0, beta1, r_squared)."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    beta1 = cov_xy / var_x if var_x != 0 else 0.0
    beta0 = mean_y - beta1 * mean_x

    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (beta0 + beta1 * xi)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return beta0, beta1, r_squared


class TwoStageLeastSquaresIV:
    def fit(self, z: List[float], x: List[float], y: List[float]) -> Dict[str, float]:
        """
        Estimate causal effect via 2SLS:
        Stage 1: Regress X on Z -> X_hat
        Stage 2: Regress Y on X_hat -> beta_iv
        """
        n = len(z)
        # Stage 1
        gamma0, gamma1, r2_stage1 = simple_linear_regression(z, x)
        x_hat = [gamma0 + gamma1 * zi for zi in z]

        # Stage 1 F-statistic: F = (R^2 / 1) / ((1 - R^2) / (n - 2))
        f_stat = (r2_stage1 / (1.0 - r2_stage1)) * (n - 2) if r2_stage1 < 1.0 else 999.0

        # Stage 2
        beta0, beta_iv, _ = simple_linear_regression(x_hat, y)

        # Naive OLS benchmark
        _, beta_ols, _ = simple_linear_regression(x, y)

        return {
            "beta_iv": beta_iv,
            "beta_ols": beta_ols,
            "stage1_f_stat": f_stat,
            "stage1_r2": r2_stage1,
            "is_strong_instrument": f_stat > 10.0
        }
