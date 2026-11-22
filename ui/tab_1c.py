import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m1c_congestion as m1c
from gt.params import DomainParams
from ui.style import (AQUA, BLUE, INK, INK_2, MUTED, ORANGE, SEQ_BLUE, YELLOW, card, chart, how_to_read,
                      module_header, show_badge)

# Drawing positions for the corridor (schematic, not to scale).
NODE_XY = {"O": (0, 0), "A": (1, 1), "B": (1, -1), "C": (1, -2.1), "D": (2, 0)}
NODE_NAME = {"O": "Maharagama", "A": "Nugegoda", "B": "Rajagiriya", "C": "Kirulapana", "D": "Fort (CBD)"}
LABEL_POS = {"O": "middle left", "A": "top center", "B": "bottom left", "C": "bottom center", "D": "middle right"}


@st.cache_data(show_spinner=False)
def _flows(d: float, fly: bool, cap: float):
    edges = m1c.network(fly, cap)
    ne, so = m1c.wardrop(edges, d), m1c.social_optimum(edges, d)
    return edges, m1c.paths(edges), ne, so, m1c.relative_gap(edges, ne)


@st.cache_data(show_spinner="Solving 40 demand levels...")
def _sweep(cap: float):
    ds = np.linspace(100, 5000, 40)
    return ds, m1c.sweep(ds, cap)


def _delay_colour(ratio: float) -> str:
    """Sequential blue by delay ratio t/t0: light = free-flowing, dark = jammed."""
    idx = int(np.clip((ratio - 1) / 1.5 * (len(SEQ_BLUE) - 1), 0, len(SEQ_BLUE) - 1))
    return SEQ_BLUE[max(idx, 2)]


def _network_figure(edges, r: m1c.FlowResult, title: str) -> go.Figure:
    fig = go.Figure()
    xmax = max(r.edge_flow.max(), 1)
    for e, x, t in zip(edges, r.edge_flow, r.edge_time):
        (x0, y0), (x1, y1) = NODE_XY[e.tail], NODE_XY[e.head]
        used = x > 0.5
        fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="text",
                        text=f"<b>{e.name}</b><br>{x:,.0f} vehicles/h<br>{t:.1f} min (free flow {e.t0:.0f})",
                        line=dict(width=2 + 14 * x / xmax if used else 1.5,
                                  color=_delay_colour(t / e.t0) if used else "#d7d6cf",
                                  dash="solid" if used else "dot"))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, showarrow=False, bgcolor="rgba(255,255,255,0.92)",
                           bordercolor="#e1e0d9", borderwidth=1, borderpad=3,
                           text=f"<b>{x:,.0f}</b>/h · {t:.1f} min" if used else "unused",
                           font=dict(size=10, color=INK if used else MUTED))
    fig.add_scatter(x=[v[0] for v in NODE_XY.values()], y=[v[1] for v in NODE_XY.values()],
                    mode="markers+text", text=[NODE_NAME[k] for k in NODE_XY],
                    textposition=[LABEL_POS[k] for k in NODE_XY], textfont=dict(size=12, color=INK),
                    marker=dict(size=18, color=INK, line=dict(color="#ffffff", width=3)), showlegend=False,
                    hoverinfo="skip")
    fig.update_layout(title=title, xaxis=dict(visible=False, range=[-0.75, 2.6]),
                      yaxis=dict(visible=False, range=[-2.6, 1.5]), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render(p: DomainParams) -> None:
    module_header("1C", "Congestion and the Price of Anarchy",
                  "Drivers travel from Maharagama to Fort and each takes the fastest route. How much time does "
                  "selfish routing waste, and can a new flyover make traffic worse?",
                  ["Congestion game", "Wardrop equilibrium", "Social optimum", "Price of Anarchy", "Braess's paradox"],
                  "Wijerathne R. M. N. S.")
    c1, c2 = st.columns([3, 1])
    d = c1.slider("Drivers per hour", 100, 5000, 1500, 50, key="1c_d")
    c2.write("")
    fly = c2.toggle("Open the new flyover", value=True, key="1c_fly")

    edges, P, ne, so, gap = _flows(float(d), fly, p.cap_scale)
    poa = ne.total_cost / so.total_cost
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selfish average trip", f"{ne.total_cost / d:.1f} min")
    k2.metric("Optimal average trip", f"{so.total_cost / d:.1f} min",
              delta=f"{(so.total_cost - ne.total_cost) / d:+.1f} min", delta_color="inverse")
    k3.metric("Price of Anarchy", f"{poa:.3f}", help="Selfish total travel time divided by the optimum.")
    k4.metric("Wardrop gap", f"{gap:.1e}", help="Close to 0 means no driver can save time by switching route.")

    with card("Road network", f"{len(edges)} road segments. Line width shows traffic; darker blue means more delay "
                              "relative to free flow. Times and capacities are illustrative."):
        c1, c2 = st.columns(2)
        with c1:
            chart(_network_figure(edges, ne, "Selfish routing (Wardrop equilibrium)"), 400, key="1c_ne")
        with c2:
            chart(_network_figure(edges, so, "Central planner (social optimum)"), 400, key="1c_so")
        st.dataframe(pd.DataFrame({
            "Route": [" → ".join([NODE_NAME[edges[pth[0]].tail]] + [NODE_NAME[edges[i].head] for i in pth])
                      for pth in P],
            "Selfish drivers/h": ne.path_flow.round(0).astype(int), "Selfish time (min)": ne.path_time.round(1),
            "Optimal drivers/h": so.path_flow.round(0).astype(int), "Optimal time (min)": so.path_time.round(1),
        }), hide_index=True, width="stretch")
        st.latex(r"t_e(x_e)=t^0_e\left[1+0.15\left(\frac{x_e}{k_e}\right)^4\right],\quad "
                 r"C(f)=\sum_e x_e t_e(x_e),\quad PoA(d)=\frac{C(f^{NE})}{C(f^{OPT})}")

    ds, sw = _sweep(p.cap_scale)
    braess = sw["ne_fly"] > sw["ne_base"] * (1 + 1e-4)
    with card("Across the day", "Every demand level from off-peak to rush hour, with and without the flyover."):
        if braess.any():
            lo, hi = ds[braess].min(), ds[braess].max()
            show_badge(None, f"Braess's paradox between about {lo:,.0f} and {hi:,.0f} drivers per hour")
        else:
            show_badge(True, "No Braess's paradox at this road capacity")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=ds, y=sw["poa_base"], name="Without flyover", line=dict(color=BLUE, width=2),
                            hovertemplate="%{x:,.0f} drivers/h<br>PoA %{y:.3f}<extra>Without flyover</extra>")
            fig.add_scatter(x=ds, y=sw["poa_fly"], name="With flyover", line=dict(color=ORANGE, width=2),
                            hovertemplate="%{x:,.0f} drivers/h<br>PoA %{y:.3f}<extra>With flyover</extra>")
            fig.add_hline(y=4 / 3, line=dict(dash="dash", color=INK_2, width=1),
                          annotation_text="4/3 bound for linear roads", annotation_font_color=INK_2)
            fig.add_vline(x=d, line=dict(color=MUTED, width=1, dash="dot"))
            fig.update_layout(title="Price of Anarchy", xaxis_title="Drivers per hour", yaxis_title="PoA",
                              xaxis_tickformat=",", hovermode="x unified")
            chart(fig, 380, key="1c_poa")
        with c2:
            fig = go.Figure()
            if braess.any():
                fig.add_vrect(x0=lo, x1=hi, fillcolor=YELLOW, opacity=0.12, line_width=0,
                              annotation_text="Flyover makes trips longer", annotation_position="top left",
                              annotation_font_color=INK_2)
            fig.add_scatter(x=ds, y=sw["ne_base"] / ds, name="Selfish, no flyover", line=dict(color=BLUE, width=2))
            fig.add_scatter(x=ds, y=sw["ne_fly"] / ds, name="Selfish, with flyover", line=dict(color=ORANGE, width=2))
            fig.add_scatter(x=ds, y=sw["opt_fly"] / ds, name="Optimum, with flyover",
                            line=dict(color=AQUA, width=2, dash="dot"))
            fig.add_vline(x=d, line=dict(color=MUTED, width=1, dash="dot"))
            fig.update_layout(title="Average trip time", xaxis_title="Drivers per hour", yaxis_title="Minutes",
                              xaxis_tickformat=",", hovermode="x unified")
            fig.update_traces(hovertemplate="%{y:.1f} min")
            chart(fig, 380, key="1c_trip")
        how_to_read("The dotted vertical line marks the demand chosen above. In the shaded band, opening the flyover "
                    "lures every driver onto the same shortcut, which jams the two narrow roads it joins, so the "
                    "average trip gets longer even though a road was added.")
