import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m1b_extensive as m1b
from gt.params import DomainParams
from ui.style import AQUA, AXIS, BLUE, INK, MUTED, ORANGE, card, chart, how_to_read, module_header

PLAYER_COLOUR = {"Platform": AQUA, "F1": BLUE, "F2": ORANGE}


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
            chosen = prob > 1e-9
            (x0, y0), (x1, y1) = pos[n.id], pos[ch.id]
            fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                            line=dict(width=4 if chosen else 1.5, color=INK if chosen else AXIS))
            label = a if prob in (0, 1) else f"{a} {prob:.0%}"
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label, showarrow=False,
                               font=dict(size=11, color=INK if chosen else MUTED), bgcolor="rgba(255,255,255,0.9)")
    first = True
    for iset, ids in stats["infosets"].items():
        if len(ids) > 1:
            fig.add_scatter(x=[pos[i][0] for i in ids], y=[pos[i][1] for i in ids], mode="lines",
                            line=dict(dash="dash", width=2, color=ORANGE), name="Information set (F2 cannot tell)",
                            showlegend=first, hoverinfo="skip")
            first = False
    for player, colour in PLAYER_COLOUR.items():
        nodes = [n for n in m1b.walk(root) if n.player == player]
        fig.add_scatter(x=[pos[n.id][0] for n in nodes], y=[pos[n.id][1] for n in nodes], mode="markers",
                        name=player, marker=dict(size=18, color=colour, line=dict(color="#ffffff", width=2)),
                        hovertext=[f"{player} decides at {n.id}" for n in nodes], hoverinfo="text")
    leaves = [n for n in m1b.walk(root) if n.terminal]
    fig.add_scatter(x=[pos[n.id][0] for n in leaves], y=[pos[n.id][1] for n in leaves], mode="markers+text",
                    text=["<br>".join(f"{v / 1000:.1f}k" for v in n.payoff) for n in leaves],
                    textposition="bottom center", textfont=dict(size=11, color=INK),
                    name="Outcome (F1, F2, Platform)",
                    hovertext=[f"{n.id}<br>F1 Rs {n.payoff[0]:,.0f}<br>F2 Rs {n.payoff[1]:,.0f}"
                               f"<br>Platform Rs {n.payoff[2]:,.0f}" for n in leaves], hoverinfo="text",
                    marker=dict(size=12, symbol="square", color=[INK if n.id in on_path else "#ffffff" for n in leaves],
                                line=dict(color=INK, width=1.5)))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-4.9, 0.4]),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render(p: DomainParams) -> None:
    module_header("1B", "Surge and entry: extensive form and SPNE",
                  "A concert ends and an event zone needs rides. The platform sets surge, F1 decides to enter, F2 "
                  "decides without seeing F1, and if both enter the platform decides on a bonus. Which plan is "
                  "credible at every step?",
                  ["Game tree", "Information sets", "Subgame perfection", "Generalized backward induction"],
                  "Vishmani P. C.")
    with st.expander("Scenario settings", icon=":material/tune:", expanded=False):
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        t = m1b.TreeParams(
            sigma_high=c1.slider("Surge multiplier when surging σ_H", 1.1, 3.0, 1.5, 0.1, key="1b_sig"),
            epsilon=c2.slider("Demand drop per unit of surge ε", 0.0, 0.5, 0.3, 0.05, key="1b_eps"),
            p0=c3.slider("Standard fare p₀ (Rs)", 500, 1200, 750, 50, key="1b_p0"),
            e=c4.slider("Cost to reposition a fleet e (Rs)", 0, 60000, 15000, 1000, key="1b_e"),
            r0=c5.slider("Profit from staying home r₀ (Rs/h)", 0, 150000, 30000, 1000, key="1b_r0"),
            bonus=c6.slider("Driver bonus b (Rs/ride)", 0, 300, 50, 10, key="1b_b"),
        )
        st.latex(r"Q_\sigma=Q(1-\varepsilon(\sigma-1)),\quad R_i=\min\{Q_\sigma/\#\text{entrants},K\},\quad "
                 r"\pi_i=((1-\kappa)\sigma p_0+b-c)R_i-e,\quad \pi_P=(\kappa\sigma p_0-b)\textstyle\sum R_i")
    root, strategy, value, order, stats = _solve(p, t)
    surge = max(strategy["P:surge"], key=strategy["P:surge"].get)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Platform chooses", surge)
    k2.metric("F1 profit in SPNE", f"Rs {value[0]:,.0f}")
    k3.metric("F2 profit in SPNE", f"Rs {value[1]:,.0f}")
    k4.metric("Platform profit in SPNE", f"Rs {value[2]:,.0f}")

    with card("Game tree", f"{stats['terminal_nodes']} outcomes · {stats['decision_levels']} decision levels · "
                           f"{stats['imperfect_info_sets']} information sets. Dark edges are the SPNE choices."):
        chart(_tree_figure(root, strategy, stats), 640, key="1b_tree")
        how_to_read("Read top to bottom. Circles are decisions (colour shows who decides). Dashed orange lines join "
                    "F2's nodes that it cannot tell apart, because it does not see F1's move. Squares are outcomes "
                    "with profits in thousands of rupees per hour for F1, F2 and the platform; filled squares are "
                    "reached in equilibrium.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        with card("SPNE strategy profile", "One choice for every information set, including ones never reached."):
            who = {"P": "Platform", "F1": "F1", "F2": "F2"}
            st.dataframe(pd.DataFrame([
                {"Decision point": s.replace("|", " after ").replace("P:", "").replace("surge", "Surge choice")
                    .replace("bonus", "Bonus"),
                 "Player": who[s.split("|")[0].split(":")[0]],
                 "Choice": ", ".join(f"{a} {pr:.0%}" if 0 < pr < 1 else a for a, pr in d.items() if pr > 1e-9)}
                for s, d in strategy.items()]), hide_index=True, width="stretch")
    with c2:
        with card("Generalized backward induction", "Smallest subgames first, folded upward to the root."):
            for k, sol in enumerate(order, 1):
                method = "best reply" if sol.kind == "perfect information" else "normal-form Nash equilibrium"
                st.markdown(f"**{k}.** `{sol.root_id}` · {method}")
            st.caption("The platform never pays the bonus: once fleets have entered, a bonus is pure cost, so a "
                       "promise to pay it is not credible.")
