"""Bregman (KL) projection of a price vector onto a constraint set.

Market-implied probabilities that violate a dependency constraint need to be
corrected to the nearest coherent vector. "Nearest" here is KL divergence
rather than Euclidean distance, which keeps the correction multiplicative and
respects that these coordinates are probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

EPS = 1e-9


def bregman_projection(
    theta: list[float],
    a_matrix: list[list[float]],
    b_vector: list[float],
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Project `theta` onto `{mu : A mu >= b, 0 < mu < 1}` under KL divergence.

    Uses the generalized (unnormalized) KL divergence
    `sum(mu*log(mu/theta) - mu + theta)`, which stays non-negative when the
    vectors do not sum to one — prices across dependent markets generally do
    not. Returns the projected vector and the divergence it cost.
    """
    theta_arr = np.clip(np.asarray(theta, dtype=float), EPS, 1.0 - EPS)
    if theta_arr.ndim != 1 or theta_arr.size == 0:
        raise ValueError("theta must be a non-empty 1-D vector")

    a_arr = np.asarray(a_matrix, dtype=float).reshape(-1, theta_arr.size)
    b_arr = np.asarray(b_vector, dtype=float)
    if a_arr.shape[0] != b_arr.size:
        raise ValueError(f"A has {a_arr.shape[0]} rows but b has {b_arr.size} entries")

    def kl_divergence(mu: np.ndarray) -> float:
        safe = np.clip(mu, EPS, None)
        return float(np.sum(safe * np.log(safe / theta_arr) - safe + theta_arr))

    def kl_gradient(mu: np.ndarray) -> np.ndarray:
        safe = np.clip(mu, EPS, None)
        return np.log(safe / theta_arr)

    constraints = [{"type": "ineq", "fun": lambda mu: a_arr @ mu - b_arr}] if b_arr.size else []
    result = minimize(
        kl_divergence,
        theta_arr.copy(),
        jac=kl_gradient,
        bounds=[(EPS, 1.0 - EPS)] * theta_arr.size,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": max_iterations},
    )

    projected = np.clip(result.x, EPS, 1.0 - EPS)
    return {
        "converged": bool(result.success),
        "message": str(result.message),
        "projected": projected.tolist(),
        "divergence": kl_divergence(projected),
        "residual": (a_arr @ projected - b_arr).tolist() if b_arr.size else [],
    }
