"""Module 1C: congestion game on a road network, Wardrop equilibrium, social optimum, PoA, Braess.

Owner: Wijerathne R. M. N. S.

Flows are continuous (vehicles per hour); travel times are in minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

ALPHA, BETA = 0.15, 4.0  # standard BPR parameters


@dataclass(frozen=True)
class Edge:
    name: str
    tail: str
    head: str
    t0: float   # free-flow time (min)
    k: float    # capacity (veh/h)


# A suburb (Maharagama) to city centre (Fort) corridor. Without the flyover there are 6 road
# segments; the flyover adds a 7th. Times and capacities are illustrative, not survey data.
BASE_EDGES = [
    Edge("High Level Rd (Maharagama-Nugegoda)", "O", "A", 8.0, 900.0),
    Edge("Nugegoda-Fort arterial", "A", "D", 22.0, 4000.0),
    Edge("Baseline outer (Maharagama-Rajagiriya)", "O", "B", 22.0, 4000.0),
    Edge("Rajagiriya-Fort link", "B", "D", 8.0, 900.0),
    Edge("Kottawa back road (Maharagama-Kirulapana)", "O", "C", 18.0, 600.0),
    Edge("Kirulapana-Fort", "C", "D", 18.0, 600.0),
]
FLYOVER = Edge("New flyover (Nugegoda-Rajagiriya)", "A", "B", 1.0, 3000.0)


def network(with_flyover: bool, cap_scale: float = 1.0) -> list[Edge]:
    """Road segments; cap_scale multiplies every capacity k_e (the shared k_e slider)."""
    edges = BASE_EDGES + ([FLYOVER] if with_flyover else [])
    return [Edge(e.name, e.tail, e.head, e.t0, e.k * cap_scale) for e in edges]


def bpr(x: np.ndarray, t0: np.ndarray, k: np.ndarray) -> np.ndarray:
    """t_e(x_e) = t0_e [1 + 0.15 (x_e / k_e)^4]."""
    return t0 * (1 + ALPHA * (x / k) ** BETA)


def bpr_integral(x, t0, k):
    return t0 * (x + ALPHA * k / (BETA + 1) * (x / k) ** (BETA + 1))


def bpr_marginal(x, t0, k):
    """d/dx [x t(x)] = t(x) + x t'(x)."""
    return t0 * (1 + (BETA + 1) * ALPHA * (x / k) ** BETA)


def paths(edges: list[Edge], origin: str = "O", dest: str = "D") -> list[list[int]]:
    out: list[list[int]] = []

    def dfs(node: str, used: list[int], seen: set[str]):
        if node == dest:
            out.append(list(used))
            return
        for i, e in enumerate(edges):
            if e.tail == node and e.head not in seen:
                dfs(e.head, used + [i], seen | {e.head})

    dfs(origin, [], {origin})
    return out


@dataclass
class FlowResult:
    path_flow: np.ndarray
    edge_flow: np.ndarray
    edge_time: np.ndarray
    path_time: np.ndarray
    total_cost: float   # vehicle-minutes per hour, C(f) = sum_e x_e t_e(x_e)


def _solve(edges: list[Edge], d: float, objective: str) -> FlowResult:
    P = paths(edges)
    M = np.zeros((len(edges), len(P)))  # edge-path incidence
    for j, pth in enumerate(P):
        M[pth, j] = 1
    t0 = np.array([e.t0 for e in edges])
    k = np.array([e.k for e in edges])

    if objective == "wardrop":   # Beckmann potential
        f_obj = lambda f: bpr_integral(M @ f, t0, k).sum()
        grad = lambda f: M.T @ bpr(M @ f, t0, k)
    else:                        # total travel time
        f_obj = lambda f: float((M @ f) @ bpr(M @ f, t0, k))
        grad = lambda f: M.T @ bpr_marginal(M @ f, t0, k)

    scale = d  # solve in shares for better conditioning
    res = minimize(lambda s: f_obj(s * scale) / scale, np.full(len(P), 1 / len(P)),
                   jac=lambda s: grad(s * scale), method="SLSQP",
                   bounds=[(0, 1)] * len(P),
                   constraints=[{"type": "eq", "fun": lambda s: s.sum() - 1, "jac": lambda s: np.ones_like(s)}],
                   options={"ftol": 1e-12, "maxiter": 500})
    f = np.clip(res.x, 0, None) * scale
    x = M @ f
    te = bpr(x, t0, k)
    return FlowResult(f, x, te, M.T @ te, float(x @ te))


def wardrop(edges: list[Edge], d: float) -> FlowResult:
    """Selfish equilibrium: every used route is a fastest route."""
    return _solve(edges, d, "wardrop")


def social_optimum(edges: list[Edge], d: float) -> FlowResult:
    return _solve(edges, d, "optimum")


def price_of_anarchy(edges: list[Edge], d: float) -> float:
    return wardrop(edges, d).total_cost / social_optimum(edges, d).total_cost


def sweep(demands: np.ndarray, cap_scale: float = 1.0) -> dict[str, np.ndarray]:
    """NE and optimum cost with and without the flyover across demand levels."""
    out = {k: [] for k in ("ne_base", "opt_base", "ne_fly", "opt_fly")}
    for d in demands:
        for tag, fly in (("base", False), ("fly", True)):
            out[f"ne_{tag}"].append(wardrop(network(fly, cap_scale), d).total_cost)
            out[f"opt_{tag}"].append(social_optimum(network(fly, cap_scale), d).total_cost)
    res = {k: np.array(v) for k, v in out.items()}
    res["poa_base"] = res["ne_base"] / res["opt_base"]
    res["poa_fly"] = res["ne_fly"] / res["opt_fly"]
    return res


def relative_gap(edges: list[Edge], r: FlowResult) -> float:
    """Wardrop check: (sum_P f_P l_P - d * min_P l_P) / (d * min_P l_P); ~0 at equilibrium."""
    d = r.path_flow.sum()
    shortest = r.path_time.min()
    return float((r.path_flow @ r.path_time - d * shortest) / (d * shortest))
