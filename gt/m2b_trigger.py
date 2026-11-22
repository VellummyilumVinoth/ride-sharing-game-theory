"""Module 2B: Grim Trigger, K-step periodic cycles, augmented stage game and the delta heatmap.

Owner: Vinoth V.

Present values follow PV_t = sum_{s>=t} delta^(s-t) pi_s (not normalised); (1 - delta) PV is the
average value plotted on the Folk Theorem polygon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .m2a_repeated import inside

DELTAS = np.round(np.arange(0.999, 0.0, -0.001), 3)  # scan from patient to impatient


def best_deviation(A: np.ndarray, B: np.ndarray, cell: tuple[int, int]) -> tuple[float, float]:
    """d_i = max over a_i' of u_i(a_i', a_-i) at the given cell."""
    i, j = cell
    return float(A[:, j].max()), float(B[i, :].max())


def grim_trigger_threshold(A, B, a_star, punish: tuple[float, float]) -> tuple[float, float, float]:
    """delta_i >= (d_i - u_i(a*)) / (d_i - e_i); 0 if a* is already a best reply for i."""
    d = best_deviation(A, B, a_star)
    u = (A[a_star], B[a_star])
    out = []
    for di, ui, ei in zip(d, u, punish):
        if di <= ui + 1e-9:
            out.append(0.0)
        elif di - ei <= 1e-9:
            out.append(float("inf"))   # punishment is no worse than the deviation payoff
        else:
            out.append((di - ui) / (di - ei))
    return out[0], out[1], max(out)


# ---------------------------------------------------------------- K-step cycles


@dataclass
class Cycle:
    cells: list[tuple[int, int]]   # the K-day sequence of action profiles
    achieved: np.ndarray           # time-average payoff of the cycle
    weights: dict[tuple[int, int], int]


def convex_weights(A: np.ndarray, B: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Weights lambda >= 0 over all cells with sum lambda u(a) = target (a vertex solution
    has at most three non-zero weights)."""
    pts = np.column_stack([A.ravel(), B.ravel()])
    n = len(pts)
    A_eq = np.vstack([pts.T, np.ones(n)])
    b_eq = np.append(target, 1.0)
    res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs-ds")
    return res.x if res.success else None


def build_cycle(A: np.ndarray, B: np.ndarray, target: np.ndarray, K: int) -> Cycle | None:
    lam = convex_weights(A, B, target)
    if lam is None:
        return None
    raw = lam * K
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - counts))[: K - counts.sum()]:   # largest remainder rounding
        counts[idx] += 1
    n_cols = A.shape[1]
    cells: list[tuple[int, int]] = []
    weights: dict[tuple[int, int], int] = {}
    for idx in np.flatnonzero(counts):
        cell = (int(idx // n_cols), int(idx % n_cols))
        weights[cell] = int(counts[idx])
        cells += [cell] * int(counts[idx])
    achieved = np.mean([[A[c], B[c]] for c in cells], axis=0)
    return Cycle(cells, achieved, weights)


def cycle_present_values(A, B, cells, delta: np.ndarray) -> np.ndarray:
    """PV_i^t = sum_{k<K} delta^k u_i(a^{(t+k) mod K}) / (1 - delta^K).

    Returns an array of shape (len(delta), K, 2).
    """
    K = len(cells)
    u = np.array([[A[c], B[c]] for c in cells])              # (K, 2)
    delta = np.atleast_1d(delta)[:, None]                       # (D, 1)
    powers = delta ** np.arange(K)                              # (D, K)
    pv = np.empty((delta.shape[0], K, 2))
    for t in range(K):
        rolled = np.roll(u, -t, axis=0)                         # u at t, t+1, ...
        pv[:, t, :] = powers @ rolled
    return pv / (1 - delta[:, :, None] ** K)


def cycle_holds(A, B, cells, delta: np.ndarray, punish: tuple[float, float]) -> np.ndarray:
    """One-shot deviation check in the augmented stage game at every position of the cycle.

    Conform:  u_i(a^t) + delta PV_i^{t+1}
    Deviate:  max_{a_i'} u_i(a_i', a^t_-i) + delta e_i / (1 - delta)
    Returns a boolean array over delta.
    """
    delta = np.atleast_1d(delta)
    pv = cycle_present_values(A, B, cells, delta)              # PV^t
    K = len(cells)
    ok = np.ones(len(delta), dtype=bool)
    for t, cell in enumerate(cells):
        conform = pv[:, t, :]                                    # = u(a^t) + delta PV^{t+1}
        d = np.array(best_deviation(A, B, cell))
        deviate = d[None, :] + (delta[:, None] * np.array(punish)[None, :]) / (1 - delta[:, None])
        ok &= (conform >= deviate - 1e-9).all(axis=1)
    return ok


def min_delta(A, B, cells, punish) -> float:
    """Smallest delta such that the cycle passes for every delta' >= delta on the 0.001 grid."""
    ok = cycle_holds(A, B, cells, DELTAS, punish)
    if not ok[0]:
        return float("nan")
    fail = np.flatnonzero(~ok)
    return float(DELTAS[fail[0] - 1]) if len(fail) else float(DELTAS[-1])


def augmented_matrix(A, B, cells, t: int, delta: float, punish) -> tuple[np.ndarray, np.ndarray]:
    """Augmented stage game at position t: U_i(z | s*, h) = u_i(z) + delta PV_{i,t+1}(h, z, s*).

    On-path cell keeps the cycle going; any other cell triggers the punishment for ever. For the
    one-shot deviation test only unilateral deviations matter, so every off-path cell uses e_i.
    """
    K = len(cells)
    pv_next = cycle_present_values(A, B, cells, np.array([delta]))[0, (t + 1) % K]
    cont_punish = np.array(punish) / (1 - delta)
    UA = A + delta * cont_punish[0]
    UB = B + delta * cont_punish[1]
    on = cells[t]
    UA[on] = A[on] + delta * pv_next[0]
    UB[on] = B[on] + delta * pv_next[1]
    return UA, UB


def delta_heatmap(A, B, Vstar: np.ndarray, mu: tuple[float, float], punish, K: int, grid: int = 50):
    """Minimum delta for each target average payoff on a grid x grid lattice over V*."""
    if len(Vstar) < 3:
        return None
    xs = np.linspace(Vstar[:, 0].min(), Vstar[:, 0].max(), grid)
    ys = np.linspace(Vstar[:, 1].min(), Vstar[:, 1].max(), grid)
    H = np.full((grid, grid), np.nan)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            v = np.array([x, y])
            if not inside(Vstar, v, strict_floor=mu):
                continue
            cyc = build_cycle(A, B, v, K)
            if cyc is not None:
                H[iy, ix] = min_delta(A, B, cyc.cells, punish)
    return xs, ys, H
