import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m1a_normal_form as m1a
from gt.params import DomainParams
from ui.common import bimatrix_frame, mix_text, owner


@st.cache_data(show_spinner=False)
def _solve(p: DomainParams, m: int):
    g = m1a.build_payoffs(p, m)
    rows, cols, steps = m1a.iesds(g.A, g.B)
    eqs, checked = m1a.support_enumeration(g.A, g.B)
    return g, rows, cols, steps, eqs, checked


@st.cache_data(show_spinner=False)
def _zone(Q: float, n: int, r1: int, r2: int, seed: int):
    zg = m1a.build_zone_game(Q, n, r1, r2, seed)
    return zg, m1a.solve_zero_sum(zg.Z)


def render(p: DomainParams) -> None:
    owner("Kulathunga W. P. M. W.")
    st.markdown("Two fleets choose a fare level at the same time. Payoffs are profits in Rs per hour.")
    m = st.slider("Number of fare levels m", 3, 6, 3, key="1a_m")
    g, rows, cols, steps, eqs, checked = _solve(p, m)

    st.markdown("**Payoff matrix (F1 , F2)**")
    st.dataframe(bimatrix_frame(g.A, g.B, g.labels))
    st.caption("Fares after surge: " + ", ".join(f"{l} Rs {f:,.0f}" for l, f in zip(g.labels, g.fares)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**IESDS (strict dominance)**")
        if not steps:
            st.write("No strictly dominated fare. IESDS leaves the game unchanged.")
        for k, s in enumerate(steps, 1):
            st.write(f"{k}. F{s.player} removes **{g.labels[s.removed]}** "
                     f"(strictly dominated by {g.labels[s.dominated_by]})")
        st.write("Surviving: F1 " + ", ".join(g.labels[r] for r in rows) + " | F2 " + ", ".join(g.labels[c] for c in cols))
    with c2:
        st.markdown(f"**Nash equilibria by support enumeration** ({checked} support pairs checked)")
        st.dataframe(pd.DataFrame([
            {"Type": "pure" if e.is_pure else "mixed", "F1": mix_text(e.p, g.labels),
             "F2": mix_text(e.q, g.labels), "u1 (Rs/h)": f"{e.u1:,.0f}", "u2 (Rs/h)": f"{e.u2:,.0f}"}
            for e in eqs]), hide_index=True)

    st.markdown("**Expected-payoff envelope**")
    st.caption("F1 mixes over the chosen fares. Each line (or plane) is F2's expected payoff from one pure reply; "
               "the upper envelope is F2's best reply. F1's equilibrium mix makes F2 indifferent where the "
               "supported replies meet.")
    picks = st.multiselect("F1 mixes over (choose 2 for 2D, 3 for 3D)", g.labels, default=g.labels[:2], key="1a_env",
                           max_selections=3)
    idx = [g.labels.index(x) for x in picks]
    if len(idx) == 2:
        pgrid, vals = m1a.envelope_2d(g.B, tuple(idx))
        fig = go.Figure()
        for r in range(vals.shape[1]):
            fig.add_scatter(x=pgrid, y=vals[:, r], mode="lines", name=f"F2 plays {g.labels[r]}")
        fig.add_scatter(x=pgrid, y=vals.max(axis=1), mode="lines", name="Upper envelope",
                        line=dict(width=5, color="black", dash="dot"))
        fig.update_layout(xaxis_title=f"Probability F1 plays {picks[0]} (rest {picks[1]})",
                          yaxis_title="F2 expected payoff (Rs/h)", height=420)
        st.plotly_chart(fig)
    elif len(idx) == 3:
        P1, P2, surfaces = m1a.envelope_3d(g.B, tuple(idx))
        fig = go.Figure()
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for r, Zr in enumerate(surfaces):
            colour = palette[r % len(palette)]
            fig.add_surface(x=P1, y=P2, z=Zr, name=f"F2 plays {g.labels[r]}", showscale=False, opacity=0.75,
                            colorscale=[[0, colour], [1, colour]], showlegend=True)
        fig.update_layout(scene=dict(xaxis_title=f"P({picks[0]})", yaxis_title=f"P({picks[1]})",
                                     zaxis_title="F2 payoff"), height=560)
        st.plotly_chart(fig)
    else:
        st.info("Pick two or three fares.")

    st.divider()
    st.markdown("**Zero-sum zone-positioning game (linear program)**")
    st.caption("Each fleet parks its idle cars in one pickup zone. A rider goes to the nearer fleet that can "
               "reach the zone. F1's payoff is its market-share lead, so F2 gets the negative.")
    z1, z2, z3, z4 = st.columns(4)
    n = z1.slider("Zones", 3, 10, 7, key="1a_zones")
    r1 = z2.slider("F1 reach (zones)", 0, 4, 1, key="1a_r1")
    r2 = z3.slider("F2 reach (zones)", 0, 4, 2, key="1a_r2")
    seed = z4.number_input("Demand seed", 0, 999, 7, key="1a_seed")
    zg, (pz, qz, v) = _zone(p.Q, n, r1, r2, int(seed))
    st.latex(r"\max_{p,\,w}\ w\quad\text{s.t.}\quad \sum_i p_i Z_{ij}\ge w\ \ \forall j,\quad \sum_i p_i=1,\ p\ge 0")
    st.metric("Value of the game (F1 share lead)", f"{v:+.3f}")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(z=zg.Z, x=zg.zones, y=zg.zones, colorscale="RdBu", zmid=0,
                                   colorbar=dict(title="F1 lead")))
        fig.update_layout(xaxis_title="F2 zone", yaxis_title="F1 zone", height=420)
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_bar(x=zg.zones, y=pz, name="F1 maxmin mix")
        fig.add_bar(x=zg.zones, y=qz, name="F2 minmax mix")
        fig.add_scatter(x=zg.zones, y=zg.demand / zg.demand.max(), name="Demand (scaled)", mode="lines+markers")
        fig.update_layout(barmode="group", yaxis_title="Probability", height=420)
        st.plotly_chart(fig)
