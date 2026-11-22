"""Module 1B: extensive-form surge/entry game and generalized backward induction (SPNE).

Owner: Vishmani P. C.

The solver works on any finite tree built from `Node`s, so the tree in `build_tree` can be
changed (more stages, players or information sets) without touching `solve_spne`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .m1a_normal_form import pure_nash, support_enumeration
from .params import DomainParams

PLAYERS = ("F1", "F2", "Platform")


@dataclass
class Node:
    id: str
    player: str | None = None            # None for a terminal node
    infoset: str | None = None           # nodes sharing an infoset are indistinguishable
    children: dict[str, "Node"] = field(default_factory=dict)
    payoff: tuple[float, ...] | None = None  # (F1, F2, Platform) at terminals

    @property
    def terminal(self) -> bool:
        return not self.children


# ---------------------------------------------------------------- domain tree


@dataclass(frozen=True)
class TreeParams:
    sigma_high: float = 1.5   # surge multiplier when the platform chooses Surge
    epsilon: float = 0.3      # demand drop per unit of surge above 1
    p0: float = 750.0         # standard fare (Rs)
    e: float = 15000.0        # cost of repositioning a fleet into the event zone (Rs)
    r0: float = 30000.0       # profit a fleet earns by staying in its home area (Rs/h)
    bonus: float = 50.0       # driver bonus per ride when the platform pays it (Rs)


def build_tree(p: DomainParams, t: TreeParams) -> Node:
    """Platform sets surge -> F1 enters or stays -> F2 (not seeing F1) enters or stays ->
    if both enter, the platform decides whether to pay a driver bonus.
    """
    def fleet_profit(sigma: float, b: float, rides: float) -> float:
        return ((1 - p.kappa) * sigma * t.p0 + b - p.c) * rides - t.e

    def leaf(name: str, sigma: float, f1_in: bool, f2_in: bool, b: float) -> Node:
        q_sigma = p.Q * (1 - t.epsilon * (sigma - 1))
        entrants = int(f1_in) + int(f2_in)
        rides = min(q_sigma / entrants, p.K) if entrants else 0.0
        u1 = fleet_profit(sigma, b, rides) if f1_in else t.r0
        u2 = fleet_profit(sigma, b, rides) if f2_in else t.r0
        up = (p.kappa * sigma * t.p0 - b) * rides * entrants
        return Node(name, payoff=(u1, u2, up))

    root = Node("root", "Platform", "P:surge")
    for s_name, sigma in (("Normal", 1.0), ("Surge", t.sigma_high)):
        f1 = Node(f"{s_name}", "F1", f"F1|{s_name}")
        root.children[s_name] = f1
        for a1 in ("Enter", "Stay"):
            f2 = Node(f"{s_name}/{a1}", "F2", f"F2|{s_name}")  # same infoset for both F1 moves
            f1.children[a1] = f2
            for a2 in ("Enter", "Stay"):
                name = f"{s_name}/{a1}/{a2}"
                if a1 == "Enter" and a2 == "Enter":
                    pb = Node(name, "Platform", f"P:bonus|{s_name}")
                    pb.children["Bonus"] = leaf(name + "/Bonus", sigma, True, True, t.bonus)
                    pb.children["No bonus"] = leaf(name + "/No bonus", sigma, True, True, 0.0)
                    f2.children[a2] = pb
                else:
                    f2.children[a2] = leaf(name, sigma, a1 == "Enter", a2 == "Enter", 0.0)
    return root


# ---------------------------------------------------------------- tree helpers


def walk(node: Node):
    yield node
    for ch in node.children.values():
        yield from walk(ch)


def tree_stats(root: Node) -> dict:
    nodes = list(walk(root))
    infosets: dict[str, list[str]] = {}
    for n in nodes:
        if not n.terminal:
            infosets.setdefault(n.infoset, []).append(n.id)

    def depth(n: Node) -> int:
        return 0 if n.terminal else 1 + max(depth(c) for c in n.children.values())

    return {
        "terminal_nodes": sum(n.terminal for n in nodes),
        "decision_levels": depth(root),
        "imperfect_info_sets": sum(len(v) > 1 for v in infosets.values()),
        "infosets": infosets,
    }


# ---------------------------------------------------------------- generalized backward induction


@dataclass
class SubgameSolution:
    root_id: str
    kind: str                         # "perfect information" or "normal form"
    choices: dict[str, dict[str, float]]  # infoset -> {action: probability}
    value: tuple[float, ...]


def _infoset_nodes(root: Node) -> dict[str, list[Node]]:
    out: dict[str, list[Node]] = {}
    for n in walk(root):
        if not n.terminal:
            out.setdefault(n.infoset, []).append(n)
    return out


def _is_subgame_root(node: Node, all_sets: dict[str, list[Node]]) -> bool:
    """A subgame starts at one node, keeps every later move and splits no information set."""
    inside = {id(n) for n in walk(node)}
    if len(all_sets[node.infoset]) > 1:
        return False
    for n in walk(node):
        if not n.terminal and any(id(m) not in inside for m in all_sets[n.infoset]):
            return False
    return True


def _expected(node: Node, plan: dict[str, dict[str, float]]) -> np.ndarray:
    if node.terminal:
        return np.array(node.payoff, dtype=float)
    dist = plan[node.infoset]
    return sum(prob * _expected(node.children[a], plan) for a, prob in dist.items() if prob > 0)


def _solve_subgame(node: Node, players: tuple[str, ...]) -> SubgameSolution:
    sets = _infoset_nodes(node)
    if all(len(v) == 1 for v in sets.values()):
        # perfect information inside: ordinary backward induction
        plan: dict[str, dict[str, float]] = {}

        def bi(n: Node) -> np.ndarray:
            if n.terminal:
                return np.array(n.payoff, dtype=float)
            vals = {a: bi(c) for a, c in n.children.items()}
            k = players.index(n.player)
            best = max(vals, key=lambda a: vals[a][k])
            plan[n.infoset] = {best: 1.0}
            return vals[best]

        v = bi(node)
        return SubgameSolution(node.id, "perfect information", plan, tuple(v))

    # imperfect information: convert to normal form over each player's information sets
    movers = sorted({n.player for v in sets.values() for n in v}, key=players.index)
    pure: dict[str, list[dict[str, str]]] = {}
    for pl in movers:
        own = [s for s, v in sets.items() if v[0].player == pl]
        acts = [list(sets[s][0].children) for s in own]
        pure[pl] = [dict(zip(own, combo)) for combo in product(*acts)]

    def outcome(profile: dict[str, dict[str, str]]) -> np.ndarray:
        plan = {s: {a: 1.0} for strat in profile.values() for s, a in strat.items()}
        return _expected(node, plan)

    if len(movers) == 2:
        p1, p2 = movers
        k1, k2 = players.index(p1), players.index(p2)
        S1, S2 = pure[p1], pure[p2]
        U = [[outcome({p1: s1, p2: s2}) for s2 in S2] for s1 in S1]
        A = np.array([[u[k1] for u in row] for row in U])
        B = np.array([[u[k2] for u in row] for row in U])
        eqs, _ = support_enumeration(A, B)
        if not eqs:
            raise ValueError(f"no equilibrium found in subgame {node.id}")
        # prefer a pure equilibrium, then the one with the highest joint payoff of the movers
        eq = max(eqs, key=lambda e: (e.is_pure, e.u1 + e.u2))
        plan: dict[str, dict[str, float]] = {}
        for strats, probs in ((S1, eq.p), (S2, eq.q)):
            for s_idx, pr in enumerate(probs):
                for iset, act in strats[s_idx].items():
                    plan.setdefault(iset, {})
                    plan[iset][act] = plan[iset].get(act, 0.0) + float(pr)
        value = sum(eq.p[i] * eq.q[j] * U[i][j] for i in range(len(S1)) for j in range(len(S2)))
        return SubgameSolution(node.id, "normal form", plan, tuple(value))

    # more than two movers: search pure equilibria of the normal form
    profiles = list(product(*[pure[pl] for pl in movers]))
    for prof in profiles:
        prof_d = dict(zip(movers, prof))
        u = outcome(prof_d)
        stable = True
        for pl in movers:
            k = players.index(pl)
            for alt in pure[pl]:
                if outcome({**prof_d, pl: alt})[k] > u[k] + 1e-9:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            plan = {s: {a: 1.0} for strat in prof_d.values() for s, a in strat.items()}
            return SubgameSolution(node.id, "normal form", plan, tuple(u))
    raise ValueError(f"no pure equilibrium in subgame {node.id}")


def solve_spne(root: Node, players: tuple[str, ...] = PLAYERS):
    """Generalized backward induction.

    1. find valid subgame roots, 2. solve the smallest ones (best reply at single nodes,
    normal-form NE where an information set is involved), 3. replace each solved subgame by
    its equilibrium payoff, 4. repeat up to the root.
    Returns (strategy per infoset, root value, list of solved subgames in order).
    """
    import copy

    work = copy.deepcopy(root)
    strategy: dict[str, dict[str, float]] = {}
    order: list[SubgameSolution] = []
    while not work.terminal:
        sets = _infoset_nodes(work)
        roots = [n for n in walk(work) if not n.terminal and _is_subgame_root(n, sets)]
        # smallest subgames: roots with no other subgame root strictly below them
        root_ids = {id(n) for n in roots}
        minimal = [n for n in roots if not any(id(m) in root_ids for m in walk(n) if m is not n)]
        for n in minimal:
            sol = _solve_subgame(n, players)
            strategy.update(sol.choices)
            order.append(sol)
            n.children = {}
            n.payoff = sol.value
            n.player = None
    return strategy, work.payoff, order


def equilibrium_path(root: Node, strategy: dict[str, dict[str, float]]) -> list[str]:
    """Node ids reached with positive probability under the SPNE."""
    reached = []

    def go(n: Node):
        reached.append(n.id)
        if n.terminal:
            return
        for a, pr in strategy[n.infoset].items():
            if pr > 1e-9:
                go(n.children[a])

    go(root)
    return reached
