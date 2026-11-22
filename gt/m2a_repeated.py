"""Module 2A: repeated-game geometry (minmax floors, feasible set, Folk Theorem region).

Owner: Karunarathna W. K.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .m1a_normal_form import BimatrixGame, build_payoffs
from .params import DomainParams


def stage_game(p: DomainParams, m: int = 3) -> BimatrixGame:
    """The 1A fare game used as the stage game (m = 3 or 4)."""
    return build_payoffs(p, m)


def pure_minmax(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """m_i = min over rival actions of i's best-response payoff."""
    m1 = A.max(axis=0).min()   # F2 picks a column to hold F1 down
    m2 = B.max(axis=1).min()   # F1 picks a row to hold F2 down
    return float(m1), float(m2)


def mixed_minmax(A: np.ndarray, B: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """mu_i = min over rival mixes q of max_a sum_b q(b) u_i(a, b), solved as an LP.

    Returns (mu_1, mu_2, punishing mix used by F2 against F1, punishing mix used by F1 against F2).
    """
    def lp(M: np.ndarray):
        # M[own action, rival action]; variables (q_1..q_n, w); minimise w
        n_own, n_riv = M.shape
        c = np.zeros(n_riv + 1)
        c[-1] = 1
        A_ub = np.hstack([M, -np.ones((n_own, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n_own), A_eq=[[1] * n_riv + [0]], b_eq=[1],
                      bounds=[(0, None)] * n_riv + [(None, None)], method="highs")
        return float(res.x[-1]), np.clip(res.x[:-1], 0, None)

    mu1, q_vs1 = lp(A)
    mu2, p_vs2 = lp(B.T)
    return mu1, mu2, q_vs1, p_vs2


def payoff_points(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.column_stack([A.ravel(), B.ravel()])


def feasible_polygon(points: np.ndarray) -> np.ndarray:
    """Vertices (counter-clockwise) of V = conv{u(a)}."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:  # all points on a line
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        return pts[[order[0], order[-1]]]


def clip_halfplane(poly: np.ndarray, axis: int, bound: float) -> np.ndarray:
    """Keep the part of a convex polygon with v[axis] >= bound (Sutherland-Hodgman)."""
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cin, nin = cur[axis] >= bound, nxt[axis] >= bound
        if cin:
            out.append(cur)
        if cin != nin:
            t = (bound - cur[axis]) / (nxt[axis] - cur[axis])
            out.append(cur + t * (nxt - cur))
    return np.array(out) if out else np.empty((0, 2))


def folk_region(V: np.ndarray, mu: tuple[float, float]) -> np.ndarray:
    """V* = {v in V | v_i > mu_i}; returned as the closed polygon used for plotting."""
    poly = clip_halfplane(V, 0, mu[0])
    if len(poly):
        poly = clip_halfplane(poly, 1, mu[1])
    if len(poly) > 1:  # clipping through a vertex repeats it
        keep = [i for i in range(len(poly)) if not np.allclose(poly[i], poly[i - 1])]
        poly = poly[keep]
    return poly


def inside(poly: np.ndarray, pt: np.ndarray, strict_floor: tuple[float, float] | None = None) -> bool:
    """Point-in-convex-polygon test (counter-clockwise vertices)."""
    if len(poly) < 3:
        return False
    if strict_floor is not None and (pt[0] <= strict_floor[0] or pt[1] <= strict_floor[1]):
        return False
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) < -1e-9:
            return False
    return True
