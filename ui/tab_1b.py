import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m1b_extensive as m1b
from gt.params import DomainParams
from ui.common import owner


@st.cache_data(show_spinner=False)
def _solve(p: DomainParams, t: m1b.TreeParams):
    root = m1b.build_tree(p, t)
    strategy, value, order = m1b.solve_spne(root)
    return root, strategy, value, order, m1b.tree_stats(root)


def _layout(root: m1b.Node):
    """x by leaf order, y by depth; parents sit above the middle of their children."""
    pos, leaves = {}, [0]

    def place(n, depth):
        if n.terminal:
            pos[n.id] = (leaves[0], -depth)
            leaves[0] += 1
        else:
            xs = [place(c, depth + 1) for c in n.children.values()]
            pos[n.id] = (sum(xs) / len(xs), -depth)
        return pos[n.id][0]

    place(root, 0)
    return pos


def _tree_figure(root, strategy, stats) -> go.Figure:
    pos = _layout(root)
    on_path = set(m1b.equilibrium_path(root, strategy))
    fig = go.Figure()
    for n in m1b.walk(root):
        for a, ch in n.children.items():
            prob = strategy.get(n.infoset, {}).get(a, 0)
            (x0, y0), (x1, y1) = pos[n.id], pos[ch.id]
            fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                            line=dict(width=5 if prob > 1e-9 else 1.2, color="#d62728" if prob > 1e-9 else "#999"))
            label = a if prob in (0, 1) else f"{a} ({prob:.2f})"
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label, showarrow=False,
                               font=dict(size=10), bgcolor="rgba(255,255,255,0.7)")
    for iset, ids in stats["infosets"].items():
        if len(ids) > 1:
            xs = [pos[i][0] for i in ids]
            ys = [pos[i][1] for i in ids]
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(dash="dash", width=2, color="#1f77b4"),
                            name="Information set", showlegend=iset.endswith("Normal"))
    dec = [n for n in m1b.walk(root) if not n.terminal]
    fig.add_scatter(x=[pos[n.id][0] for n in dec], y=[pos[n.id][1] for n in dec], mode="markers+text",
                    text=[n.player for n in dec], textposition="top center", name="Decision node",
                    marker=dict(size=14, color="#1f77b4"))
    leaves = [n for n in m1b.walk(root) if n.terminal]
    fig.add_scatter(x=[pos[n.id][0] for n in leaves], y=[pos[n.id][1] for n in leaves], mode="markers+text",
                    text=["<br>".join(f"{v / 1000:.1f}k" for v in n.payoff) for n in leaves],
                    textposition="bottom center", name="Terminal (F1, F2, Platform)",
                    marker=dict(size=10, color=["#d62728" if n.id in on_path else "#2ca02c" for n in leaves],
                                symbol="square"))
    fig.update_layout(height=620, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h"), margin=dict(t=20, b=80))
    return fig


def render(p: DomainParams) -> None:
    owner("Vishmani P. C.")
    st.markdown("A concert ends and an event zone needs rides. The platform sets surge, fleet F1 decides to enter "
                "or stay, fleet F2 decides without seeing F1, and if both enter the platform decides on a bonus.")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    t = m1b.TreeParams(
        sigma_high=c1.slider("Surge σ_H", 1.1, 3.0, 1.5, 0.1, key="1b_sig"),
        epsilon=c2.slider("Demand drop ε", 0.0, 0.5, 0.3, 0.05, key="1b_eps"),
        p0=c3.slider("Standard fare p₀ (Rs)", 500, 1200, 750, 50, key="1b_p0"),
        e=c4.slider("Repositioning cost e (Rs)", 0, 60000, 15000, 1000, key="1b_e"),
        r0=c5.slider("Home-area profit r₀ (Rs/h)", 0, 150000, 30000, 1000, key="1b_r0"),
        bonus=c6.slider("Bonus b (Rs/ride)", 0, 300, 50, 10, key="1b_b"),
    )
    root, strategy, value, order, stats = _solve(p, t)
    st.latex(r"Q_\sigma=Q(1-\varepsilon(\sigma-1)),\quad R_i=\min\{Q_\sigma/\#\text{entrants},K\},\quad "
             r"\pi_i=((1-\kappa)\sigma p_0+b-c)R_i-e,\quad \pi_P=(\kappa\sigma p_0-b)\textstyle\sum R_i")

    m1, m2, m3 = st.columns(3)
    m1.metric("Terminal nodes", stats["terminal_nodes"])
    m2.metric("Decision levels", stats["decision_levels"])
    m3.metric("Imperfect-information sets", stats["imperfect_info_sets"])

    st.plotly_chart(_tree_figure(root, strategy, stats))
    st.caption("Red edges are SPNE choices; blue dashed lines join nodes in the same information set. "
               "Terminal labels are payoffs in thousands of Rs per hour (F1, F2, Platform).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SPNE strategy profile (one action per information set)**")
        st.dataframe(pd.DataFrame([
            {"Information set": s, "Player": s.split("|")[0].split(":")[0].replace("P", "Platform"),
             "Choice": ", ".join(f"{a} ({pr:.2f})" if 0 < pr < 1 else a for a, pr in d.items() if pr > 1e-9)}
            for s, d in strategy.items()]), hide_index=True)
        st.metric("Equilibrium payoffs (F1, F2, Platform)", " , ".join(f"{v:,.0f}" for v in value))
    with c2:
        st.markdown("**Generalized backward induction order**")
        for k, sol in enumerate(order, 1):
            st.write(f"{k}. Subgame at `{sol.root_id}` solved by {sol.kind}, value "
                     f"({', '.join(f'{v:,.0f}' for v in sol.value)})")
