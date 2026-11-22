import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m2b_trigger as m2b
from gt.m2a_repeated import inside
from gt.params import DomainParams
from ui.common import mix_text, owner
from ui.tab_2a import geometry, polygon_figure


@st.cache_data(show_spinner="Computing the minimum δ on a 50 × 50 grid...")
def _heatmap(A, B, Vstar, mu, punish, K):
    return m2b.delta_heatmap(A, B, Vstar, mu, punish, K, grid=50)


def render(p: DomainParams) -> None:
    owner("Vinoth V.")
    m = st.radio("Stage game size", [3, 4], horizontal=True, format_func=lambda k: f"{k} × {k}", key="2b_m")
    g, m_pure, mu, _, V, Vstar, eqs = geometry(p, m)
    A, B, L = g.A, g.B, g.labels
    delta = p.delta
    cells = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])]
    joint = tuple(int(x) for x in np.unravel_index((A + B).argmax(), A.shape))

    st.markdown("#### Punishment")
    c1, c2 = st.columns(2)
    kind = c1.radio("After a deviation, both fleets play", ["Nash reversion (SPNE)", "Mixed minmax μ (NE only)"],
                    key="2b_kind")
    if kind.startswith("Nash"):
        ne_sorted = sorted(eqs, key=lambda e: e.u1 + e.u2)  # harshest credible punishment first
        pick = c2.selectbox("Stage Nash equilibrium used as punishment", range(len(ne_sorted)), key="2b_ne",
                            format_func=lambda k: f"F1 {mix_text(ne_sorted[k].p, L)} | F2 {mix_text(ne_sorted[k].q, L)}"
                                                  f"  ({ne_sorted[k].u1:,.0f}, {ne_sorted[k].u2:,.0f})")
        punish = (ne_sorted[pick].u1, ne_sorted[pick].u2)
    else:
        punish = mu
        c2.write(f"Each fleet is held to its mixed minmax: e = ({mu[0]:,.0f}, {mu[1]:,.0f}). A minmax threat for "
                 "ever is a Nash equilibrium of the repeated game, not a subgame perfect one.")

    st.markdown("#### Grim Trigger")
    a_star = st.selectbox("Collusive profile a*", cells, index=cells.index(joint), key="2b_astar",
                          format_func=lambda c: f"({L[c[0]]}, {L[c[1]]})  u = ({A[c]:,.0f}, {B[c]:,.0f})")
    st.latex(r"\frac{u_i(a^*)}{1-\delta}\ \ge\ d_i+\frac{\delta e_i}{1-\delta}\iff "
             r"\delta\ \ge\ \frac{d_i-u_i(a^*)}{d_i-e_i}")
    d1, d2 = m2b.best_deviation(A, B, a_star)
    t1, t2, t = m2b.grim_trigger_threshold(A, B, a_star, punish)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best deviation d₁ (Rs/h)", f"{d1:,.0f}")
    c2.metric("Best deviation d₂ (Rs/h)", f"{d2:,.0f}")
    c3.metric("δ₁ needed", "never" if np.isinf(t1) else f"{t1:.3f}")
    c4.metric("δ₂ needed", "never" if np.isinf(t2) else f"{t2:.3f}")
    c5.metric(f"Sustained at δ = {delta:.2f}?", "Yes" if delta >= t else "No")

    st.markdown("#### K-step periodic cycle for an interior target")
    if len(Vstar) < 3:
        st.warning("V* is empty or degenerate for these parameters, so no interior target exists.")
        return
    lo, hi = Vstar.min(axis=0), Vstar.max(axis=0)
    centre = Vstar.mean(axis=0)
    c1, c2, c3 = st.columns(3)
    v1 = c1.slider("Target v₁ (Rs/h)", float(lo[0]), float(hi[0]), float(centre[0]), key="2b_v1")
    v2 = c2.slider("Target v₂ (Rs/h)", float(lo[1]), float(hi[1]), float(centre[1]), key="2b_v2")
    K = c3.slider("Cycle length K (days)", 1, 12, 6, key="2b_K")
    target = np.array([v1, v2])
    if not inside(Vstar, target, strict_floor=mu):
        st.warning("That target is outside V*. Move it into the green region.")
        return
    cyc = m2b.build_cycle(A, B, target, K)
    if cyc is None:
        st.warning("No cycle reaches this target.")
        return
    dmin = m2b.min_delta(A, B, cyc.cells, punish)
    st.latex(r"PV_i^t=\frac{\sum_{k=0}^{K-1}\delta^k u_i\!\left(a^{(t+k)\bmod K}\right)}{1-\delta^K},\qquad "
             r"U_i(z\mid s^*,h)=u_i(z)+\delta\,PV_{i,t+1}(\{h,z\},s^*)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cycle average payoff", f"({cyc.achieved[0]:,.0f}, {cyc.achieved[1]:,.0f})",
              help="Close to the target; the gap comes from rounding the weights to whole days. A larger K gets closer.")
    c2.metric("Minimum δ for this cycle", "not sustainable" if np.isnan(dmin) else f"{dmin:.3f}")
    c3.metric(f"Sustained at δ = {delta:.2f}?", "No" if np.isnan(dmin) or delta < dmin else "Yes")
    st.dataframe(pd.DataFrame([{"Day": t + 1, "F1": L[c[0]], "F2": L[c[1]], "u1": f"{A[c]:,.0f}", "u2": f"{B[c]:,.0f}"}
                               for t, c in enumerate(cyc.cells)]), hide_index=True)

    with st.expander("Augmented stage game at one position of the cycle"):
        pos = st.slider("Position t", 1, K, 1, key="2b_pos") - 1
        UA, UB = m2b.augmented_matrix(A, B, cyc.cells, pos, delta, punish)
        on = cyc.cells[pos]
        st.dataframe(pd.DataFrame([[f"{UA[i, j]:,.0f} , {UB[i, j]:,.0f}" for j in range(len(L))] for i in range(len(L))],
                                  index=[f"F1: {l}" for l in L], columns=[f"F2: {l}" for l in L]))
        ok = UA[on] >= UA[:, on[1]].max() - 1e-6 and UB[on] >= UB[on[0], :].max() - 1e-6
        st.write(f"On-path cell ({L[on[0]]}, {L[on[1]]}) is {'a' if ok else 'NOT a'} Nash equilibrium of this "
                 f"augmented game at δ = {delta:.2f}.")

    days = np.arange(1, 61)
    seq = [cyc.cells[(t - 1) % K] for t in days]
    pay = np.array([[A[c], B[c]] for c in seq])
    fig = go.Figure()
    fig.add_scatter(x=days, y=pay[:, 0], name="F1 daily payoff", mode="lines+markers", line=dict(shape="hv"))
    fig.add_scatter(x=days, y=pay[:, 1], name="F2 daily payoff", mode="lines+markers", line=dict(shape="hv"))
    w = delta ** (days - 1)
    fig.add_scatter(x=days, y=np.cumsum(w[:, None] * pay, axis=0)[:, 0] / np.cumsum(w), name="F1 discounted average",
                    line=dict(dash="dot"))
    fig.update_layout(title="Payoff path of the cycle", xaxis_title="Day", yaxis_title="Rs/h", height=360)
    st.plotly_chart(fig)

    st.markdown("#### Minimum patience heatmap over V*")
    st.caption("Each grid point is a target average payoff. Its cycle is built from at most three cells with "
               "K-day rounding, and δ is scanned down from 0.999 in steps of 0.001 until a deviation pays.")
    hm = _heatmap(A, B, Vstar, mu, tuple(punish), K)
    if hm is not None:
        xs, ys, H = hm

        def add(fig):
            fig.add_trace(go.Heatmap(x=xs, y=ys, z=H, colorscale="Viridis", zmin=np.nanmin(H) if np.isfinite(H).any() else 0,
                                     zmax=1, colorbar=dict(title="min δ"), hovertemplate="v1 %{x:,.0f}<br>v2 %{y:,.0f}<br>min δ %{z:.3f}<extra></extra>"))
            fig.add_scatter(x=[target[0]], y=[target[1]], mode="markers", name="Chosen target",
                            marker=dict(size=14, color="white", line=dict(color="black", width=2)))

        st.plotly_chart(polygon_figure(g, m_pure, mu, V, Vstar, eqs, extra=add))
