"""Module 1A: normal-form fare game, IESDS, support enumeration and the zero-sum zone LP.

Owner: Kulathunga W. P. M. W.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from .params import DomainParams, fare_levels

TOL = 1e-9


# ---------------------------------------------------------------- fare game


@dataclass
class BimatrixGame:
    labels: list[str]
    fares: np.ndarray  # actual fares after surge (Rs)
    A: np.ndarray      # fleet F1 payoffs, rows = F1 action, cols = F2 action
    B: np.ndarray      # fleet F2 payoffs


def demand(Q: float, beta: float, a_own: float, a_other: float) -> float:
    """D_i = min{Q, max{0, Q/2 - beta (a_i - a_j)}}."""
    return min(Q, max(0.0, Q / 2 - beta * (a_own - a_other)))


def build_payoffs(p: DomainParams, m: int = 3) -> BimatrixGame:
    """u_i(a) = ((1 - kappa) a_i - c) min{D_i(a), K} for every pair of fare levels."""
    labels, base = fare_levels(m)
    fares = base * p.sigma
    A = np.zeros((m, m))
    B = np.zeros((m, m))
    for i, ai in enumerate(fares):
        for j, aj in enumerate(fares):
            A[i, j] = ((1 - p.kappa) * ai - p.c) * min(demand(p.Q, p.beta, ai, aj), p.K)
            B[i, j] = ((1 - p.kappa) * aj - p.c) * min(demand(p.Q, p.beta, aj, ai), p.K)
    return BimatrixGame(labels, fares, A, B)


# ---------------------------------------------------------------- IESDS


@dataclass
class EliminationStep:
    player: int          # 1 or 2
    removed: int         # action index removed
    dominated_by: int    # action index that strictly dominates it


def iesds(A: np.ndarray, B: np.ndarray) -> tuple[list[int], list[int], list[EliminationStep]]:
    """Iterated elimination of strictly dominated (pure) strategies.

    Returns the surviving row and column indices and the elimination steps in order.
    """
    rows = list(range(A.shape[0]))
    cols = list(range(A.shape[1]))
    steps: list[EliminationStep] = []
    changed = True
    while changed:
        changed = False
        for r in rows:
            for r2 in rows:
                if r2 != r and all(A[r2, c] > A[r, c] + TOL for c in cols):
                    rows.remove(r)
                    steps.append(EliminationStep(1, r, r2))
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        for c in cols:
            for c2 in cols:
                if c2 != c and all(B[r, c2] > B[r, c] + TOL for r in rows):
                    cols.remove(c)
                    steps.append(EliminationStep(2, c, c2))
                    changed = True
                    break
            if changed:
                break
    return rows, cols, steps


# ---------------------------------------------------------------- Nash equilibria


@dataclass
class MixedNE:
    p: np.ndarray   # fleet 1 mixed strategy
    q: np.ndarray   # fleet 2 mixed strategy
    u1: float
    u2: float

    @property
    def is_pure(self) -> bool:
        return bool(np.isclose(self.p.max(), 1) and np.isclose(self.q.max(), 1))


def _solve_indifference(M: np.ndarray, own: tuple, other: tuple) -> tuple[np.ndarray, float] | None:
    """Mix x over `other` so that every action in `own` earns the same payoff w under M.

    M is the payoff matrix of the player choosing `own` (rows) against `other` (cols).
    Returns (x over all columns, w) or None if no valid probability vector exists.
    """
    k = len(own)
    # unknowns: x_other (k), w ; equations: M[own, other] x - w = 0 (k), sum x = 1
    lhs = np.zeros((k + 1, k + 1))
    lhs[:k, :k] = M[np.ix_(own, other)]
    lhs[:k, k] = -1
    lhs[k, :k] = 1
    rhs = np.zeros(k + 1)
    rhs[k] = 1
    sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if not np.allclose(lhs @ sol, rhs, atol=1e-7):
        return None
    x_sub, w = sol[:k], sol[k]
    if (x_sub < -1e-9).any():
        return None
    x = np.zeros(M.shape[1])
    x[list(other)] = np.clip(x_sub, 0, None)
    x /= x.sum()
    return x, w


def support_enumeration(A: np.ndarray, B: np.ndarray) -> tuple[list[MixedNE], int]:
    """All Nash equilibria with equal-size supports (complete for non-degenerate games).

    Returns the equilibria and the number of support pairs checked.
    """
    m, n = A.shape
    found: list[MixedNE] = []
    checked = 0
    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                checked += 1
                # q over J makes fleet 1 indifferent over I; p over I makes fleet 2 indifferent over J
                sq = _solve_indifference(A, I, J)
                sp = _solve_indifference(B.T, J, I)
                if sq is None or sp is None:
                    continue
                q, w1 = sq
                p, w2 = sp
                # no action outside the support may pay more
                if (A @ q > w1 + 1e-7).any() or (p @ B > w2 + 1e-7).any():
                    continue
                if any(np.allclose(p, e.p, atol=1e-6) and np.allclose(q, e.q, atol=1e-6) for e in found):
                    continue
                found.append(MixedNE(p, q, float(p @ A @ q), float(p @ B @ q)))
    return found, checked


def pure_nash(A: np.ndarray, B: np.ndarray) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(A.shape[0])
        for j in range(A.shape[1])
        if A[i, j] >= A[:, j].max() - TOL and B[i, j] >= B[i, :].max() - TOL
    ]


# ---------------------------------------------------------------- zero-sum zone game


@dataclass
class ZoneGame:
    zones: list[str]
    demand: np.ndarray  # riders per hour waiting in each zone
    Z: np.ndarray       # F1 market-share lead (F2 receives -Z)


def build_zone_game(Q: float, n_zones: int = 7, reach1: int = 1, reach2: int = 2, seed: int = 7) -> ZoneGame:
    """Each fleet parks its idle cars in one of n pickup zones on a corridor.

    A rider is served by the nearer fleet if that fleet can reach the zone (reach r zones);
    ties are split. F1's payoff is its market-share lead over F2, so the game is zero-sum.
    """
    rng = np.random.default_rng(seed)
    centre = (n_zones - 1) / 2
    peak = np.exp(-(((np.arange(n_zones) - centre) / max(1.0, n_zones / 4)) ** 2))
    weights = peak * rng.uniform(0.7, 1.3, n_zones)
    dem = Q * weights / weights.sum()
    Z = np.zeros((n_zones, n_zones))
    for i in range(n_zones):
        for j in range(n_zones):
            s1 = s2 = 0.0
            for z in range(n_zones):
                d1, d2 = abs(z - i), abs(z - j)
                ok1, ok2 = d1 <= reach1, d2 <= reach2
                if ok1 and (not ok2 or d1 < d2):
                    s1 += dem[z]
                elif ok2 and (not ok1 or d2 < d1):
                    s2 += dem[z]
                elif ok1 and ok2:
                    s1 += dem[z] / 2
                    s2 += dem[z] / 2
            Z[i, j] = (s1 - s2) / Q
    return ZoneGame([f"Zone {z + 1}" for z in range(n_zones)], dem, Z)


def solve_zero_sum(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Maxmin strategy of the row player and minmax strategy of the column player via LP.

    Row LP: max w s.t. sum_i p_i Z[i, j] >= w for all j, sum p = 1, p >= 0.
    """
    m, n = Z.shape
    # variables (p_1..p_m, w); minimise -w
    c = np.zeros(m + 1)
    c[-1] = -1
    A_ub = np.hstack([-Z.T, np.ones((n, 1))])  # w - sum_i p_i Z_ij <= 0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=[[1] * m + [0]], b_eq=[1],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    p, value = res.x[:m], res.x[-1]
    # column player: min v s.t. sum_j q_j Z[i, j] <= v for all i
    c2 = np.zeros(n + 1)
    c2[-1] = 1
    A_ub2 = np.hstack([Z, -np.ones((m, 1))])
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(m), A_eq=[[1] * n + [0]], b_eq=[1],
                   bounds=[(0, None)] * n + [(None, None)], method="highs")
    q = res2.x[:n]
    return np.clip(p, 0, None), np.clip(q, 0, None), float(value)


# ---------------------------------------------------------------- payoff envelopes


def envelope_2d(M_opponent: np.ndarray, mixer_actions: tuple[int, int], grid: int = 201):
    """Opponent's expected payoff of each pure reply while the mixer mixes two actions.

    M_opponent[mixer_action, reply] holds the opponent's payoff. Returns p (weight on the
    first mixer action) and a (grid x replies) array.
    """
    a, b = mixer_actions
    p = np.linspace(0, 1, grid)
    vals = np.outer(p, M_opponent[a]) + np.outer(1 - p, M_opponent[b])
    return p, vals


def envelope_3d(M_opponent: np.ndarray, mixer_actions: tuple[int, int, int], grid: int = 41):
    """Same as envelope_2d over the probability simplex of three mixer actions."""
    a, b, c = mixer_actions
    s = np.linspace(0, 1, grid)
    P1, P2 = np.meshgrid(s, s)
    P3 = 1 - P1 - P2
    mask = P3 < -1e-12
    surfaces = []
    for r in range(M_opponent.shape[1]):
        Zr = P1 * M_opponent[a, r] + P2 * M_opponent[b, r] + P3 * M_opponent[c, r]
        Zr = np.where(mask, np.nan, Zr)
        surfaces.append(Zr)
    return P1, P2, surfaces
