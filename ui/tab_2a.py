import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gt import m2a_repeated as m2a
from gt.m1a_normal_form import support_enumeration
from gt.params import DomainParams
from ui.common import bimatrix_frame, mix_text
from ui.style import AXIS, BLUE, INK, INK_2, MUTED, ORANGE, RED, card, chart, how_to_read, module_header


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


def polygon_figure(g, m_pure, mu, V, Vstar, eqs, extra=None, show_labels=True) -> go.Figure:
    fig = go.Figure()
    if len(V) >= 3:
        loop = np.vstack([V, V[:1]])
        fig.add_scatter(x=loop[:, 0], y=loop[:, 1], fill="toself", fillcolor="rgba(195,194,183,0.22)",
                        line=dict(color=AXIS, width=1.5), name="Feasible set V", hoverinfo="skip")
    if len(Vstar) >= 3:
        loop = np.vstack([Vstar, Vstar[:1]])
        fig.add_scatter(x=loop[:, 0], y=loop[:, 1], fill="toself", fillcolor="rgba(42,120,214,0.16)",
                        line=dict(color=BLUE, width=2), name="Folk region V*", hoverinfo="skip")
    if extra:
        extra(fig)
    pts = [(g.A[i, j], g.B[i, j], f"{g.labels[i]} / {g.labels[j]}")
           for i in range(g.A.shape[0]) for j in range(g.A.shape[1])]
    fig.add_scatter(x=[x for x, _, _ in pts], y=[y for _, y, _ in pts],
                    mode="markers+text" if show_labels else "markers",
                    text=[t for _, _, t in pts], textposition="top right", textfont=dict(size=10, color=INK_2),
                    marker=dict(size=8, color=INK, line=dict(color="#ffffff", width=2)), name="Fare pairs (F1 / F2)",
                    hovertemplate="%{text}<br>F1 Rs %{x:,.0f}<br>F2 Rs %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=[e.u1 for e in eqs], y=[e.u2 for e in eqs], mode="markers", name="Stage Nash equilibria",
                    marker=dict(size=16, symbol="star", color=ORANGE, line=dict(color="#ffffff", width=2)),
                    hovertemplate="Stage NE<br>F1 Rs %{x:,.0f}<br>F2 Rs %{y:,.0f}<extra></extra>")
    joint = np.unravel_index((g.A + g.B).argmax(), g.A.shape)
    fig.add_scatter(x=[g.A[joint]], y=[g.B[joint]], mode="markers", name="Collusive (joint-profit) outcome",
                    marker=dict(size=16, symbol="diamond", color=RED, line=dict(color="#ffffff", width=2)),
                    hovertemplate="Collusion<br>F1 Rs %{x:,.0f}<br>F2 Rs %{y:,.0f}<extra></extra>")
    fig.add_vline(x=m_pure[0], line=dict(dash="dash", color=INK_2, width=1), annotation_text="m₁",
                  annotation_font_color=INK_2)
    fig.add_hline(y=m_pure[1], line=dict(dash="dash", color=INK_2, width=1), annotation_text="m₂",
                  annotation_font_color=INK_2)
    if not np.isclose(mu[0], m_pure[0]) or not np.isclose(mu[1], m_pure[1]):
        fig.add_vline(x=mu[0], line=dict(dash="dot", color=MUTED, width=1), annotation_text="μ₁",
                      annotation_position="bottom right")
        fig.add_hline(y=mu[1], line=dict(dash="dot", color=MUTED, width=1), annotation_text="μ₂",
                      annotation_position="bottom left")
    fig.update_layout(xaxis_title="F1 average profit (Rs/h)", yaxis_title="F2 average profit (Rs/h)",
                      xaxis_tickformat=",", yaxis_tickformat=",", legend=dict(y=1.02))
    return fig


def render(p: DomainParams) -> None:
    module_header("2A", "Repeated fare game: payoff geometry",
                  "The fare game is played every day. What is the least each fleet can be held to, and which "
                  "average profits could a long-run agreement deliver?",
                  ["Repeated game", "Pure minmax", "Mixed minmax (LP)", "Feasible set", "Folk Theorem"],
                  "Karunarathna W. K.")
    m = st.segmented_control("Stage game", [3, 4], default=3, key="2a_m",
                             format_func=lambda k: f"{k} × {k} ({'Low, Standard, Surge' if k == 3 else '+ Premium'})") or 3
    g, m_pure, mu, (q_vs1, p_vs2), V, Vstar, eqs = geometry(p, m)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pure minmax m₁", f"Rs {m_pure[0]:,.0f}")
    k2.metric("Pure minmax m₂", f"Rs {m_pure[1]:,.0f}")
    k3.metric("Mixed minmax μ₁", f"Rs {mu[0]:,.0f}")
    k4.metric("Mixed minmax μ₂", f"Rs {mu[1]:,.0f}")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        with card("Folk Theorem polygon", "Every average profit pair the fleets can reach (grey) and the ones a "
                                          "patient pair can sustain in equilibrium (blue)."):
            chart(polygon_figure(g, m_pure, mu, V, Vstar, eqs), 560, key="2a_poly")
            how_to_read("Each black dot is one fare pair played every day. Mixing days gives any point inside the "
                        "grey hull. The dashed lines are the minmax floors: no fleet accepts less, because it could "
                        "always guarantee that much. The blue region V* is what remains above both floors.")
    with c2:
        with card("Stage game", "Profit per hour (F1 · F2)."):
            st.dataframe(bimatrix_frame(g.A, g.B, g.labels), width="stretch")
        with card("How the floors are enforced"):
            st.markdown(f"F2 holds F1 to **Rs {mu[0]:,.0f}** by playing {mix_text(q_vs1, g.labels)}.  \n"
                        f"F1 holds F2 to **Rs {mu[1]:,.0f}** by playing {mix_text(p_vs2, g.labels)}.")
            st.latex(r"m_i=\min_{a_{-i}}\max_{a_i}u_i(a_i,a_{-i})")
            st.latex(r"\mu_i=\min_{q}\ w\ \ \text{s.t.}\ \sum_b q(b)\,u_i(a,b)\le w\ \ \forall a")
            st.latex(r"V^*=\{v\in V \mid v_i>\mu_i\}")
