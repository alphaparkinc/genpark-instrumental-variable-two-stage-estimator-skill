"""
Demonstration of Instrumental Variable Two-Stage Least Squares (2SLS) Estimator Skill
"""

import random
from client import TwoStageLeastSquaresIV

def main():
    print("=== Instrumental Variable (2SLS) Causal Estimation ===")
    random.seed(42)
    n = 300

    # Exogenous Instrument Z
    Z = [random.uniform(-3.0, 3.0) for _ in range(n)]
    # Unobserved Confounder U
    U = [random.uniform(-5.0, 5.0) for _ in range(n)]

    # Endogenous Treatment X influenced by Z and confounded by U
    X = [2.0 * z + 3.0 * u + random.gauss(0, 0.5) for z, u in zip(Z, U)]

    # Outcome Y with TRUE causal effect = 3.0 * X, but also confounded by 4.0 * U
    Y = [3.0 * x + 4.0 * u + random.gauss(0, 0.5) for x, u in zip(X, U)]

    estimator = TwoStageLeastSquaresIV()
    results = estimator.fit(Z, X, Y)

    print(f"Stage 1 F-Statistic: {results['stage1_f_stat']:.2f} (Strong Instrument: {results['is_strong_instrument']})")
    print(f"Naive Biased OLS Estimate:  {results['beta_ols']:.4f}")
    print(f"2SLS Causal Estimate:       {results['beta_iv']:.4f} (True Target: 3.0000)")

    assert results["is_strong_instrument"] is True
    assert 2.7 <= results["beta_iv"] <= 3.3
    assert results["beta_ols"] > 3.5  # Substantially biased upwards by confounding

    print("\nInstrumental Variable 2SLS Verification PASS!")

if __name__ == "__main__":
    main()
