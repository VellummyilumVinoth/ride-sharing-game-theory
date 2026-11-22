import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m2b_trigger as m2b
from gt.m2a_repeated import inside
from gt.params import DomainParams
from ui.common import mix_text
from ui.style import (BLUE, INK, INK_2, ORANGE, SEQ_SCALE, badge, card, chart, how_to_read, module_header,
                      show_badge)
from ui.tab_2a import geometry, polygon_figure


@st.cache_data(show_spinner="Computing the minimum δ on a 50 × 50 grid...")
def _heatmap(A, B, Vstar, mu, punish, K):
    return m2b.delta_heatmap(A, B, Vstar, mu, punish, K, grid=50)


def render(p: DomainParams) -> None:
    module_header("2B", "Collusion: trigger strategies",
                  "Both fleets would earn more if neither undercut. How patient must they be for a threat of "
                  "punishment to keep fares high every day?",
                  ["Grim Trigger", "Nash reversion", "K-step cycles", "One-shot deviation principle", "Augmented stage game"],
                  "Vinoth V.")
    m = st.segmented_control("Stage game", [3, 4], default=3, key="2b_m", format_func=lambda k: f"{k} × {k}") or 3
    g, m_pure, mu, _, V, Vstar, eqs = geometry(p, m)
    A, B, L = g.A, g.B, g.labels
    delta = p.delta
    cells = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])]
    joint = tuple(int(x) for x in np.unravel_index((A + B).argmax(), A.shape))

    with card("1 · Agreement and punishment", "Choose what the fleets agree to charge and what happens if one cheats."):
        c1, c2 = st.columns(2)
        a_star = c1.selectbox("Agreed fares a* (F1 / F2)", cells, index=cells.index(joint), key="2b_astar",
                              format_func=lambda c: f"{L[c[0]]} / {L[c[1]]}  ·  Rs {A[c]:,.0f} / Rs {B[c]:,.0f}")
        kind = c2.radio("After a deviation, both fleets play", ["Nash reversion (credible, SPNE)",
                                                                "Mixed minmax for ever (NE only)"],
                        key="2b_kind", horizontal=True)
        if kind.startswith("Nash"):
            ne_sorted = sorted(eqs, key=lambda e: e.u1 + e.u2)  # harshest credible punishment first
            pick = c2.selectbox("Stage equilibrium used as punishment", range(len(ne_sorted)), key="2b_ne",
                                format_func=lambda k: f"F1 {mix_text(ne_sorted[k].p, L)} · F2 {mix_text(ne_sorted[k].q, L)}"
                                                      f"  (Rs {ne_sorted[k].u1:,.0f} each)")
            punish = (ne_sorted[pick].u1, ne_sorted[pick].u2)
        else:
            punish = mu
            c2.caption(f"Each fleet is held to its mixed minmax, Rs {mu[0]:,.0f} / Rs {mu[1]:,.0f}. This threat "
                       "supports a Nash equilibrium of the repeated game but is not subgame perfect.")

    d1, d2 = m2b.best_deviation(A, B, a_star)
    t1, t2, t = m2b.grim_trigger_threshold(A, B, a_star, punish)
    ok = bool(delta >= t)
    with card("2 · Grim Trigger", "Cheat once and both fleets switch to the punishment for ever."):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Profit from the agreement", f"Rs {A[a_star]:,.0f}")
        k2.metric("Profit from cheating once", f"Rs {d1:,.0f}", delta=f"{d1 - A[a_star]:+,.0f} today")
        k3.metric("Patience needed", "never" if np.isinf(t) else f"δ ≥ {t:.3f}",
                  help=f"F1 needs {t1:.3f}, F2 needs {t2:.3f}.")
        k4.metric("Current patience", f"δ = {delta:.2f}")
        show_badge(ok, f"Agreement holds at δ = {delta:.2f}" if ok else f"Agreement breaks at δ = {delta:.2f}")
        st.latex(r"\frac{u_i(a^*)}{1-\delta}\ \ge\ d_i+\frac{\delta e_i}{1-\delta}\iff "
                 r"\delta\ \ge\ \frac{d_i-u_i(a^*)}{d_i-e_i}")

    if len(Vstar) < 3:
        st.warning("The Folk region V* is empty for these settings, so there is no interior target to sustain.")
        return

    with card("3 · Reach any target with a K-day cycle",
              "Profits inside V* that no single fare pair gives are reached by rotating fare pairs over K days."):
        lo, hi = Vstar.min(axis=0), Vstar.max(axis=0)
        centre = Vstar.mean(axis=0)
        c1, c2, c3 = st.columns(3)
        v1 = c1.slider("Target F1 average profit (Rs/h)", float(round(lo[0], -2)), float(round(hi[0], -2)),
                       float(round(centre[0], -2)), 100.0, format="%.0f", key="2b_v1")
        v2 = c2.slider("Target F2 average profit (Rs/h)", float(round(lo[1], -2)), float(round(hi[1], -2)),
                       float(round(centre[1], -2)), 100.0, format="%.0f", key="2b_v2")
        K = c3.slider("Cycle length K (days)", 1, 12, 6, key="2b_K")
        target = np.array([v1, v2])
        if not inside(Vstar, target, strict_floor=mu):
            show_badge(None, "That target is outside V*. Move it into the blue region.")
            return
        cyc = m2b.build_cycle(A, B, target, K)
        if cyc is None:
            show_badge(None, "No cycle reaches this target.")
            return
        dmin = m2b.min_delta(A, B, cyc.cells, punish)
        sustained = not np.isnan(dmin) and delta >= dmin
        k1, k2, k3 = st.columns(3)
        k1.metric("Cycle average (F1 / F2)", f"{cyc.achieved[0]:,.0f} / {cyc.achieved[1]:,.0f}",
                  help="Close to the target; the gap comes from rounding to whole days. A larger K gets closer.")
        k2.metric("Patience needed for this cycle", "not sustainable" if np.isnan(dmin) else f"δ ≥ {dmin:.3f}")
        k3.markdown("<br>" + badge(sustained, "Cycle holds" if sustained else "Cycle breaks"), unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.dataframe(pd.DataFrame([{"Day": t + 1, "F1 fare": L[c[0]], "F2 fare": L[c[1]],
                                        "F1 profit": f"{A[c]:,.0f}", "F2 profit": f"{B[c]:,.0f}"}
                                       for t, c in enumerate(cyc.cells)]), hide_index=True, width="stretch")
        with c2:
            days = np.arange(1, 4 * K + 1)
            seq = [cyc.cells[(t - 1) % K] for t in days]
            pay = np.array([[A[c], B[c]] for c in seq])
            w = delta ** (days - 1)
            fig = go.Figure()
            fig.add_scatter(x=days, y=pay[:, 0], name="F1 daily profit", mode="lines+markers",
                            line=dict(shape="hv", color=BLUE, width=2), marker=dict(size=8))
            fig.add_scatter(x=days, y=pay[:, 1], name="F2 daily profit", mode="lines+markers",
                            line=dict(shape="hv", color=ORANGE, width=2), marker=dict(size=8))
            fig.add_scatter(x=days, y=np.cumsum(w * pay[:, 0]) / np.cumsum(w), name="F1 discounted average",
                            line=dict(dash="dot", color=INK_2, width=2))
            fig.update_layout(title="Profit over four cycles", xaxis_title="Day", yaxis_title="Rs per hour",
                              yaxis_tickformat=",", hovermode="x unified")
            chart(fig, 330, key="2b_path")
        st.latex(r"PV_i^t=\frac{\sum_{k=0}^{K-1}\delta^k u_i\!\left(a^{(t+k)\bmod K}\right)}{1-\delta^K},\qquad "
                 r"U_i(z\mid s^*,h)=u_i(z)+\delta\,PV_{i,t+1}(\{h,z\},s^*)")

        with st.expander("One-shot deviation check: augmented stage game on a chosen day", icon=":material/fact_check:"):
            pos = st.slider("Day of the cycle", 1, K, 1, key="2b_pos") - 1
            UA, UB = m2b.augmented_matrix(A, B, cyc.cells, pos, delta, punish)
            on = cyc.cells[pos]
            st.dataframe(pd.DataFrame([[f"{UA[i, j]:,.0f}  ·  {UB[i, j]:,.0f}" for j in range(len(L))]
                                       for i in range(len(L))],
                                      index=[f"F1 {l}" for l in L], columns=[f"F2 {l}" for l in L]), width="stretch")
            ne_ok = UA[on] >= UA[:, on[1]].max() - 1e-6 and UB[on] >= UB[on[0], :].max() - 1e-6
            show_badge(ne_ok, f"{L[on[0]]} / {L[on[1]]} is {'a' if ne_ok else 'not a'} Nash equilibrium of the "
                              f"augmented game at δ = {delta:.2f}")
            st.caption("Each cell is today's profit plus the discounted value of what follows: the rest of the cycle "
                       "on the agreed cell, the punishment for ever everywhere else.")

    with card("4 · Patience map over the Folk region",
              "Minimum δ for every target average profit (50 × 50 grid). Darker means more patience is needed."):
        hm = _heatmap(A, B, Vstar, mu, tuple(punish), K)
        if hm is not None:
            xs, ys, H = hm

            def add(fig):
                fig.add_trace(go.Heatmap(x=xs, y=ys, z=H, colorscale=SEQ_SCALE, zmin=0, zmax=1, name="Minimum δ",
                                         colorbar=dict(title="min δ", thickness=12, len=0.8),
                                         hovertemplate="F1 Rs %{x:,.0f}<br>F2 Rs %{y:,.0f}<br>min δ %{z:.3f}<extra></extra>"))

            fig = polygon_figure(g, m_pure, mu, V, Vstar, eqs, extra=add, show_labels=False)
            fig.add_scatter(x=[target[0]], y=[target[1]], mode="markers", name="Chosen target",
                            marker=dict(size=14, color="#ffffff", line=dict(color=INK, width=3)))
            chart(fig, 560, key="2b_heat")
            how_to_read("Each pixel is a target average profit pair inside V*. Its colour is the smallest discount "
                        "factor at which the K-day cycle reaching it survives every one-shot deviation. Targets near "
                        "the minmax floors need δ close to 1; targets near full collusion need much less patience.")
