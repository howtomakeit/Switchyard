"""Combinatorial arbitrage search over dependent prediction markets.

Given per-token prices and a payoff-cover matrix (`A z >= b`, one row per
possible state of the world), find the cheapest basket that pays out in every
state. A basket whose cost is below its guaranteed payoff is an arbitrage.

Two layers share one formulation:

* `screen_arbitrage` solves the continuous relaxation with GLOP. Its cost is a
  lower bound on the integer optimum, so a bound at or above the guaranteed
  payoff proves no arbitrage exists and the exact solve can be skipped.
* `find_arbitrage` solves the binary program exactly with SCIP and returns the
  basket to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ortools.linear_solver import pywraplp

SENSES = (">=", "<=", "==")
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Constraint:
    """One linear relation over the outcome-token selectors.

    `coeffs` is dense and positionally aligned with the price vector. A cover
    row uses `sense=">="` with `coeffs[i] = 1` for every token that pays in the
    state the row represents.
    """

    coeffs: list[float]
    b: float
    sense: str = ">="

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Constraint:
        """Parse and validate a constraint supplied by an MCP caller."""
        sense = str(raw.get("sense", ">="))
        if sense not in SENSES:
            raise ValueError(f"sense must be one of {SENSES}, got {sense!r}")
        return cls(coeffs=[float(c) for c in raw["coeffs"]], b=float(raw["b"]), sense=sense)


def _build(
    prices: list[float], constraints: list[Constraint], solver_id: str, integral: bool
) -> tuple[Any, list[Any]]:
    """Construct the shared minimum-cost-cover program for either solver."""
    if not prices:
        raise ValueError("prices must not be empty")

    solver = pywraplp.Solver.CreateSolver(solver_id)
    if solver is None:
        raise RuntimeError(f"{solver_id} solver unavailable in this OR-Tools build")

    z = [
        solver.IntVar(0, 1, f"z_{i}") if integral else solver.NumVar(0.0, 1.0, f"z_{i}")
        for i in range(len(prices))
    ]

    for index, constraint in enumerate(constraints):
        if len(constraint.coeffs) != len(prices):
            raise ValueError(
                f"constraint {index} has {len(constraint.coeffs)} coefficients "
                f"but there are {len(prices)} prices"
            )
        expr = solver.Sum(
            coeff * var for coeff, var in zip(constraint.coeffs, z, strict=True) if coeff
        )
        if constraint.sense == ">=":
            solver.Add(expr >= constraint.b)
        elif constraint.sense == "<=":
            solver.Add(expr <= constraint.b)
        else:
            solver.Add(expr == constraint.b)

    solver.Minimize(solver.Sum(price * var for price, var in zip(prices, z, strict=True)))
    return solver, z


def _guaranteed_payoff(constraints: list[Constraint], solution: list[float]) -> float | None:
    """Return the worst-case payoff of a basket across every cover row.

    Only `>=` rows describe states of the world; a problem carrying none has no
    payoff interpretation, so the result is `None` rather than a misleading 0.
    """
    cover = [c for c in constraints if c.sense == ">="]
    if not cover:
        return None
    return min(
        sum(coeff * value for coeff, value in zip(c.coeffs, solution, strict=True)) for c in cover
    )


def screen_arbitrage(prices: list[float], constraints: list[Constraint]) -> dict[str, Any]:
    """Layer 1: bound the basket cost with the LP relaxation.

    Cheap enough to run across every candidate cluster. The relaxed cost can
    only be lower than the integer optimum, so `arbitrage_possible=False` is a
    proof of absence; `True` only means the exact solve is worth running.
    """
    solver, z = _build(prices, constraints, "GLOP", integral=False)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {"status": "infeasible", "lower_bound": None, "arbitrage_possible": False}

    lower_bound = solver.Objective().Value()
    payoff = _guaranteed_payoff(constraints, [var.solution_value() for var in z])
    threshold = payoff if payoff is not None else 1.0
    return {
        "status": "optimal",
        "lower_bound": lower_bound,
        "relaxed_payoff": payoff,
        "arbitrage_possible": lower_bound < threshold - TOLERANCE,
    }


def find_arbitrage(prices: list[float], constraints: list[Constraint]) -> dict[str, Any]:
    """Layer 2: solve the binary program exactly and report the basket's edge.

    Returns the selected tokens, what the basket costs, the payoff it is
    guaranteed in the worst state, and the difference between them. A positive
    edge at these prices is an arbitrage before execution costs; whether it
    survives the order book is Layer 3's question.
    """
    solver, z = _build(prices, constraints, "SCIP", integral=True)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {
            "status": "infeasible",
            "solution": None,
            "cost": None,
            "selected": [],
            "guaranteed_payoff": None,
            "edge": None,
        }

    # SCIP returns integral values as floats; round before comparing.
    solution = [int(round(var.solution_value())) for var in z]
    cost = solver.Objective().Value()
    payoff = _guaranteed_payoff(constraints, [float(v) for v in solution])
    return {
        "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible",
        "solution": solution,
        "cost": cost,
        "selected": [i for i, picked in enumerate(solution) if picked],
        "guaranteed_payoff": payoff,
        "edge": None if payoff is None else payoff - cost,
        "is_arbitrage": payoff is not None and payoff - cost > TOLERANCE,
    }
