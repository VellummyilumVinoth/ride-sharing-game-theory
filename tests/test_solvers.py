"""Solver checks on textbook games, run before trusting the domain results.

Run with:  python -m pytest -q
"""
import numpy as np
import pytest

from gt import m1a_normal_form as m1a
from gt import m1b_extensive as m1b
from gt import m1c_congestion as m1c
from gt import m2a_repeated as m2a
from gt import m2b_trigger as m2b
from gt.params import DomainParams

PD_A = np.array([[5, 0], [6, 1.0]])  # Prisoner's Dilemma: C/D, payoffs CC=5, CD=0, DC=6, DD=1
PD_B = PD_A.T
MP_A = np.array([[1, -1], [-1, 1.0]])  # Matching Pennies


def test_iesds_prisoners_dilemma():
    rows, cols, steps = m1a.iesds(PD_A, PD_B)
    assert rows == [1] and cols == [1] and len(steps) == 2


def test_support_enumeration_matching_pennies():
    eqs, checked = m1a.support_enumeration(MP_A, -MP_A)
    assert len(eqs) == 1 and np.allclose(eqs[0].p, 0.5) and np.allclose(eqs[0].q, 0.5)
    assert checked == 5


def test_support_pairs_6x6():
    g = m1a.build_payoffs(DomainParams(), 6)
    _, checked = m1a.support_enumeration(g.A, g.B)
    assert checked == 923


def test_zero_sum_lp_rock_paper_scissors():
    Z = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0.0]])
    p, q, v = m1a.solve_zero_sum(Z)
    assert np.allclose(p, 1 / 3, atol=1e-6) and np.allclose(q, 1 / 3, atol=1e-6) and abs(v) < 1e-9


def test_default_fare_game_is_a_dilemma():
    g = m1a.build_payoffs(DomainParams(), 3)
    joint = np.unravel_index((g.A + g.B).argmax(), g.A.shape)
    assert joint not in m1a.pure_nash(g.A, g.B)


def test_tree_meets_brief():
    stats = m1b.tree_stats(m1b.build_tree(DomainParams(), m1b.TreeParams()))
    assert stats["terminal_nodes"] >= 10 and stats["decision_levels"] >= 3 and stats["imperfect_info_sets"] >= 2


def test_spne_entry_deterrence():
    # Entrant: Out (0, 2) or In; Incumbent: Fight (-1, -1) or Accommodate (1, 1). SPNE: In, Accommodate.
    root = m1b.Node("root", "F1", "E")
    inc = m1b.Node("in", "F2", "I")
    root.children = {"Out": m1b.Node("out", payoff=(0, 2, 0)), "In": inc}
    inc.children = {"Fight": m1b.Node("f", payoff=(-1, -1, 0)), "Accommodate": m1b.Node("a", payoff=(1, 1, 0))}
    strategy, value, _ = m1b.solve_spne(root)
    assert strategy["E"] == {"In": 1.0} and strategy["I"] == {"Accommodate": 1.0} and value[:2] == (1, 1)


def test_braess_paradox_on_corridor():
    # Opening the flyover makes the selfish average trip longer at 1,500 drivers per hour.
    base = m1c.wardrop(m1c.network(False), 1500).total_cost
    fly = m1c.wardrop(m1c.network(True), 1500).total_cost
    assert fly > base * 1.05
    assert m1c.price_of_anarchy(m1c.network(True), 1500) >= 1


def test_wardrop_gap_is_zero():
    edges = m1c.network(True)
    assert m1c.relative_gap(edges, m1c.wardrop(edges, 1500)) < 1e-6


def test_minmax_matching_pennies():
    assert m2a.pure_minmax(MP_A, -MP_A) == (1.0, 1.0)
    mu1, mu2, *_ = m2a.mixed_minmax(MP_A, -MP_A)
    assert abs(mu1) < 1e-9 and abs(mu2) < 1e-9


def test_grim_trigger_prisoners_dilemma():
    assert m2b.grim_trigger_threshold(PD_A, PD_B, (0, 0), (1, 1))[2] == pytest.approx(0.2)


def test_four_round_cycle_threshold():
    # Cycle (C,C),(D,D),(D,D),(D,D) with Grim punishment (D,D): delta >= (1/5)^(1/4) ~ 0.669
    cells = [(0, 0), (1, 1), (1, 1), (1, 1)]
    assert m2b.min_delta(PD_A, PD_B, cells, (1, 1)) == pytest.approx(0.2 ** 0.25, abs=1e-3)
