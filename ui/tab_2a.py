import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gt import m2a_repeated as m2a
from gt.m1a_normal_form import support_enumeration
from gt.params import DomainParams
from ui.common import bimatrix_frame, mix_text, owner


@st.cache_data(show_spinner=False)
def geometry(p: DomainParams, m: int):
    """Shared by 2A and 2B: stage game, minmax floors, V and V*."""
    g = m2a.stage_game(p, m)
    m_pure = m2a.pure_minmax(g.A, g.B)
    mu1, mu2, q_vs1, p_vs2 = m2a.mixed_minmax(g.A, g.B)
    V = m2a.feasible_polygon(m2a.payoff_points(g.A, g.B))
    Vstar = m2a.folk_region(V, (mu1, mu2))
    eqs, _ = support_enumeration(g.A, g.B)
    return g, m_pure, (mu1, mu2), (q_vs1, p_vs2), V, Vstar, eqs


def polygon_figure(g, m_pure, mu, V, Vstar, eqs, extra=None) -> go.Figure:
    fig = go.Figure()
    if len(V) >= 3:
        loop = np.vstack([V, V[:1]])
        fig.add_scatter(x=loop[:, 0], y=loop[:, 1], fill="toself", fillcolor="rgba(150,150,150,0.25)",
                        line=dict(color="grey"), name="Feasible set V")
    if len(Vstar) >= 3:
        loop = np.vstack([Vstar, Vstar[:1]])
        fig.add_scatter(x=loop[:, 0], y=loop[:, 1], fill="toself", fillcolor="rgba(44,160,44,0.35)",
                        line=dict(color="green"), name="Folk region V*")
    pts = [(g.A[i, j], g.B[i, j], f"({g.labels[i]}, {g.labels[j]})")
           for i in range(g.A.shape[0]) for j in range(g.A.shape[1])]
    fig.add_scatter(x=[x for x, _, _ in pts], y=[y for _, y, _ in pts], mode="markers+text",
                    text=[t for _, _, t in pts], textposition="top right", marker=dict(size=9, color="black"),
                    name="Pure outcomes")
    fig.add_scatter(x=[e.u1 for e in eqs], y=[e.u2 for e in eqs], mode="markers", name="Stage Nash equilibria",
                    marker=dict(size=14, symbol="star", color="gold", line=dict(color="black", width=1)))
    joint = np.unravel_index((g.A + g.B).argmax(), g.A.shape)
    fig.add_scatter(x=[g.A[joint]], y=[g.B[joint]], mode="markers", name="Joint-profit (collusive) profile",
                    marker=dict(size=16, symbol="diamond-open", color="red", line=dict(width=3)))
    fig.add_vline(x=m_pure[0], line_dash="dash", line_color="purple", annotation_text="m₁")
    fig.add_hline(y=m_pure[1], line_dash="dash", line_color="purple", annotation_text="m₂")
    fig.add_vline(x=mu[0], line_dash="dot", line_color="orange", annotation_text="μ₁", annotation_position="bottom right")
    fig.add_hline(y=mu[1], line_dash="dot", line_color="orange", annotation_text="μ₂", annotation_position="bottom left")
    if extra:
        extra(fig)
    fig.update_layout(xaxis_title="F1 average payoff (Rs/h)", yaxis_title="F2 average payoff (Rs/h)",
                      height=560, legend=dict(orientation="h", y=-0.15))
    return fig


def render(p: DomainParams) -> None:
    owner("Karunarathna W. K.")
    st.markdown("The fare game from 1A is played every day. This tab finds each fleet's security floor and the "
                "set of average payoffs the Folk Theorem can sustain.")
    m = st.radio("Stage game size", [3, 4], horizontal=True, format_func=lambda k: f"{k} × {k}", key="2a_m")
    g, m_pure, mu, (q_vs1, p_vs2), V, Vstar, eqs = geometry(p, m)
    st.dataframe(bimatrix_frame(g.A, g.B, g.labels))

    st.latex(r"m_i=\min_{a_{-i}}\max_{a_i}u_i(a_i,a_{-i}),\qquad "
             r"\mu_i=\min_{q}\ w\ \ \text{s.t.}\ \sum_b q(b)\,u_i(a,b)\le w\ \ \forall a")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pure minmax m₁", f"{m_pure[0]:,.0f}")
    c2.metric("Pure minmax m₂", f"{m_pure[1]:,.0f}")
    c3.metric("Mixed minmax μ₁", f"{mu[0]:,.0f}")
    c4.metric("Mixed minmax μ₂", f"{mu[1]:,.0f}")
    st.caption(f"F2 holds F1 to μ₁ by playing {mix_text(q_vs1, g.labels)}; "
               f"F1 holds F2 to μ₂ by playing {mix_text(p_vs2, g.labels)}.")

    st.plotly_chart(polygon_figure(g, m_pure, mu, V, Vstar, eqs))
    st.caption("V is the convex hull of all pure payoff pairs. V* keeps the points with vᵢ > μᵢ for both "
               "fleets, the payoffs a patient pair of fleets can sustain in a subgame perfect equilibrium.")
