import numpy as np
import pandas as pd
import streamlit as st

from gt import m1a_normal_form as m1a
from gt import m1b_extensive as m1b
from gt import m1c_congestion as m1c
from gt import m2b_trigger as m2b
from gt.params import DomainParams
from ui.common import TEAM
from ui.style import badge, card, hero

MODULES = [
    ("1A", "Fare game", "Two fleets set fares at once. Which fares survive, and where is the equilibrium?",
     "Kulathunga W. P. M. W."),
    ("1B", "Surge & entry", "Platform sets surge, fleets decide to enter an event zone. What is the credible plan?",
     "Vishmani P. C."),
    ("1C", "Congestion", "Every driver takes the fastest road. How much time does selfish routing waste?",
     "Wijerathne R. M. N. S."),
    ("2A", "Repeated geometry", "Fleets meet daily. Which average profits can they sustain?", "Karunarathna W. K."),
    ("2B", "Collusion", "How patient must fleets be to keep fares high without a contract?", "Vinoth V."),
    ("3–6", "Phase 2", "Learning, auctions and tolls, carpool cost sharing, collusion stress tests.", "All leads"),
]


@st.cache_data(show_spinner=False)
def _summary(p: DomainParams):
    g = m1a.build_payoffs(p, 3)
    eqs, _ = m1a.support_enumeration(g.A, g.B)
    best_ne = max(eqs, key=lambda e: e.u1 + e.u2)
    worst_ne = min(eqs, key=lambda e: e.u1 + e.u2)
    joint = tuple(int(x) for x in np.unravel_index((g.A + g.B).argmax(), g.A.shape))
    grim = m2b.grim_trigger_threshold(g.A, g.B, joint, (worst_ne.u1, worst_ne.u2))[2]
    strategy, value, _ = m1b.solve_spne(m1b.build_tree(p, m1b.TreeParams()))
    surge = max(strategy["P:surge"], key=strategy["P:surge"].get)
    d = 1500.0
    ne_fly = m1c.wardrop(m1c.network(True, p.cap_scale), d).total_cost
    ne_base = m1c.wardrop(m1c.network(False, p.cap_scale), d).total_cost
    so_fly = m1c.social_optimum(m1c.network(True, p.cap_scale), d).total_cost
    return dict(labels=g.labels, best_ne=best_ne, joint=joint, joint_u=(g.A[joint], g.B[joint]), grim=grim,
                surge=surge, spne_value=value, poa=ne_fly / so_fly, braess=ne_fly > ne_base * (1 + 1e-4),
                trip_fly=ne_fly / d, trip_base=ne_base / d)


def render(p: DomainParams) -> None:
    hero("Ride-Sharing Strategy Lab",
         "A game-theoretic simulator of competing fleets, riders and a platform in Colombo. "
         "Move the scenario controls on the left and every module recomputes.",
         ["Nash equilibrium", "IESDS", "SPNE", "Price of Anarchy", "Folk Theorem", "Trigger strategies"])

    s = _summary(p)
    L = s["labels"]
    st.markdown("##### Scenario at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Competitive profit per fleet", f"Rs {s['best_ne'].u1:,.0f}/h",
              help="Best stage Nash equilibrium of the 1A fare game.")
    c2.metric("Collusive profit per fleet", f"Rs {s['joint_u'][0]:,.0f}/h",
              delta=f"{s['joint_u'][0] - s['best_ne'].u1:+,.0f} vs competition",
              help=f"Both fleets charge {L[s['joint'][0]]}: the joint-profit maximum.")
    c3.metric("Patience needed to collude", "never" if np.isinf(s["grim"]) else f"δ ≥ {s['grim']:.2f}",
              help="Grim Trigger threshold with the harshest stage equilibrium as punishment (Module 2B).")
    c4.metric("Price of Anarchy at 1,500 cars/h", f"{s['poa']:.2f}",
              help="Selfish travel time divided by the optimum, with the flyover open (Module 1C).")

    with card("Key findings", "Recomputed for the current scenario."):
        collude_ok = not np.isinf(s["grim"]) and p.delta >= s["grim"]
        st.markdown(
            f"- {badge(collude_ok, 'Collusion sustainable' if collude_ok else 'Collusion breaks down')} "
            f"At patience δ = {p.delta:.2f}, fleets {'can' if collude_ok else 'cannot'} hold fares at "
            f"**{L[s['joint'][0]]}** with a Grim Trigger threat.\n"
            f"- {badge(None if s['braess'] else True, 'Braess paradox' if s['braess'] else 'No Braess paradox')} "
            f"At 1,500 cars per hour the average trip is **{s['trip_fly']:.1f} min** with the flyover and "
            f"**{s['trip_base']:.1f} min** without it.\n"
            f"- {badge(True, 'Credible plan')} In the surge game the platform chooses **{s['surge']}** pricing "
            f"and never pays the driver bonus once fleets have entered.",
            unsafe_allow_html=True)

    st.markdown("##### Modules")
    cols = st.columns(3)
    for k, (code, title, q, who) in enumerate(MODULES):
        with cols[k % 3]:
            st.markdown(f'<div class="gt-card"><span class="gt-code">{code}</span><h4>{title}</h4><p>{q}</p>'
                        f'<div class="gt-who">{who}</div></div>', unsafe_allow_html=True)
            st.write("")

    c1, c2 = st.columns(2)
    with c1:
        with card("Current scenario"):
            st.dataframe(pd.DataFrame([
                ("Commuter demand Q", f"{p.Q:,.0f} riders/hour"),
                ("Price sensitivity β", f"{p.beta:.1f} riders per Rs"),
                ("Surge multiplier σ", f"{p.sigma:.1f}"),
                ("Fares (10 km)", f"Rs {600 * p.sigma:,.0f} / {750 * p.sigma:,.0f} / {1000 * p.sigma:,.0f}"),
                ("Operating cost c", f"Rs {p.c:,.0f} per trip"),
                ("Commission κ", f"{p.kappa:.0%}"),
                ("Fleet capacity K", f"{p.K:,.0f} rides/hour"),
                ("Road capacity scale", f"{p.cap_scale:.1f}×"),
                ("Patience δ", f"{p.delta:.2f}"),
            ], columns=["Parameter", "Value"]), hide_index=True, width="stretch")
    with c2:
        with card("Fare payoff used by 1A, 2A and 2B", "Demand a fleet attracts, then its profit per hour."):
            st.latex(r"D_i=\min\Big\{Q,\ \max\big\{0,\ \tfrac{Q}{2}-\beta(a_i-a_j)\big\}\Big\}")
            st.latex(r"u_i(a)=\big((1-\kappa)\,a_i-c\big)\cdot\min\{D_i(a),\,K\}")
            st.caption("aᵢ is fleet i's fare, κ the commission, c the cost per trip and K the fleet's capacity.")
    with card("Team", "Phase 1 and Phase 2 ownership."):
        st.dataframe(pd.DataFrame(TEAM, columns=["Name", "Phase 1", "Phase 2"]), hide_index=True, width="stretch")
