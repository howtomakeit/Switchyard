"""Combinatorial arbitrage search over dependent prediction markets.

Given per-outcome prices and the logical constraints linking them, find the
cheapest basket of outcome tokens that satisfies every constraint. Solved as a
pure binary program with OR-Tools/SCIP: the problems are small (tens to low
hundreds of outcomes per dependency cluster) and exactness matters more than
speed at this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ortools.linear_solver import pywraplp

SENSES = (">=", "<=", "==")


@dataclass(frozen=True)
class Constraint:
    """One linear relation over the binary outcome selectors.

    `coeffs` is dense and positionally aligned with the price vector; `sense`
    and `b` give the relation, e.g. `sum(coeffs * z) == 1` for a set of
    mutually exclusive outcomes.
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
        coeffs = [float(c) for c in raw["coeffs"]]
        return cls(coeffs=coeffs, b=float(raw["b"]), sense=sense)


def find_arbitrage(prices: list[float], constraints: list[Constraint]) -> dict[str, Any]:
    """Minimize basket cost subject to the dependency constraints.

    Returns the optimal binary selection with its cost, or `status="infeasible"`
    when no basket satisfies the constraints. A cost below the guaranteed payoff
    of the constrained set is the arbitrage; deciding that is the caller's job,
    since payoff depends on which outcome resolves.
    """
    if not prices:
        raise ValueError("prices must not be empty")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver unavailable in this OR-Tools build")

    z = [solver.IntVar(0, 1, f"z_{i}") for i in range(len(prices))]

    for index, constraint in enumerate(constraints):
        if len(constraint.coeffs) != len(prices):
            raise ValueError(
                f"constraint {index} has {len(constraint.coeffs)} coefficients "
                f"but there are {len(prices)} prices"
            )
        expr = solver.Sum(coeff * var for coeff, var in zip(constraint.coeffs, z, strict=True))
        if constraint.sense == ">=":
            solver.Add(expr >= constraint.b)
        elif constraint.sense == "<=":
            solver.Add(expr <= constraint.b)
        else:
            solver.Add(expr == constraint.b)

    solver.Minimize(solver.Sum(price * var for price, var in zip(prices, z, strict=True)))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {"status": "infeasible", "solution": None, "cost": None, "selected": []}

    # SCIP returns integral values as floats; round before comparing.
    solution = [int(round(var.solution_value())) for var in z]
    return {
        "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible",
        "solution": solution,
        "cost": solver.Objective().Value(),
        "selected": [i for i, picked in enumerate(solution) if picked],
    }
