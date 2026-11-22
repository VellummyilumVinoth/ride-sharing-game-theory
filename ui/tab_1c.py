import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m1c_congestion as m1c
from gt.params import DomainParams
from ui.common import owner

# Drawing positions for the corridor (not to scale).
NODE_XY = {"O": (0, 0), "A": (1, 1), "B": (1, -1), "C": (1, -2.2), "D": (2, 0)}
NODE_NAME = {"O": "Maharagama", "A": "Nugegoda", "B": "Rajagiriya", "C": "Kirulapana", "D": "Fort (CBD)"}


@st.cache_data(show_spinner=False)
def _flows(d: float, fly: bool, cap: float):
    edges = m1c.network(fly, cap)
    ne, so = m1c.wardrop(edges, d), m1c.social_optimum(edges, d)
    return edges, m1c.paths(edges), ne, so, m1c.relative_gap(edges, ne)


@st.cache_data(show_spinner="Solving 40 demand levels...")
def _sweep(cap: float):
    ds = np.linspace(100, 5000, 40)
    return ds, m1c.sweep(ds, cap)


def _network_figure(edges, r: m1c.FlowResult, title: str) -> go.Figure:
    fig = go.Figure()
    xmax = max(r.edge_flow.max(), 1)
    for e, x, t in zip(edges, r.edge_flow, r.edge_time):
        (x0, y0), (x1, y1) = NODE_XY[e.tail], NODE_XY[e.head]
        fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="text",
                        text=f"{e.name}<br>flow {x:,.0f} veh/h<br>time {t:.1f} min",
                        line=dict(width=1 + 12 * x / xmax, color="#d62728" if t > 1.5 * e.t0 else "#1f77b4"))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"{x:,.0f} veh/h<br>{t:.1f} min",
                           showarrow=False, bgcolor="rgba(255,255,255,0.8)", font=dict(size=10))
    fig.add_scatter(x=[v[0] for v in NODE_XY.values()], y=[v[1] for v in NODE_XY.values()],
                    mode="markers+text", text=[NODE_NAME[k] for k in NODE_XY], textposition="top center",
                    marker=dict(size=16, color="#333"), showlegend=False)
    fig.update_layout(title=title, height=380, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(t=40, b=10))
    return fig


def render(p: DomainParams) -> None:
    owner("Wijerathne R. M. N. S.")
    st.markdown("Drivers travel from Maharagama to Fort and each picks the fastest route. Travel time on a "
                "road follows the BPR function, and total cost is vehicle-minutes per hour.")
    st.latex(r"t_e(x_e)=t^0_e\left[1+0.15\left(\frac{x_e}{k_e}\right)^4\right],\quad "
             r"C(f)=\sum_e x_e t_e(x_e),\quad PoA(d)=\frac{C(f^{NE})}{C(f^{OPT})}")
    c1, c2 = st.columns([3, 1])
    d = c1.slider("Drivers per hour d", 100, 5000, 1500, 50, key="1c_d")
    fly = c2.toggle("Open the flyover", value=True, key="1c_fly")

    edges, P, ne, so, gap = _flows(float(d), fly, p.cap_scale)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selfish avg trip (min)", f"{ne.total_cost / d:.2f}")
    m2.metric("Optimal avg trip (min)", f"{so.total_cost / d:.2f}")
    m3.metric("Price of Anarchy", f"{ne.total_cost / so.total_cost:.3f}")
    m4.metric("Wardrop gap", f"{gap:.1e}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(_network_figure(edges, ne, "Selfish (Wardrop) equilibrium"))
    c2.plotly_chart(_network_figure(edges, so, "Social optimum"))
    st.caption(f"{len(edges)} road segments. Red roads run at more than 1.5 times their free-flow time. "
               "Travel times and capacities are illustrative.")
    st.dataframe(pd.DataFrame({
        "Route": [" → ".join([NODE_NAME[edges[pth[0]].tail]] + [NODE_NAME[edges[i].head] for i in pth]) for pth in P],
        "Selfish flow": ne.path_flow.round(0), "Selfish time (min)": ne.path_time.round(2),
        "Optimal flow": so.path_flow.round(0), "Optimal time (min)": so.path_time.round(2),
    }), hide_index=True)

    st.divider()
    ds, sw = _sweep(p.cap_scale)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=ds, y=sw["poa_base"], name="Without flyover")
        fig.add_scatter(x=ds, y=sw["poa_fly"], name="With flyover")
        fig.add_hline(y=4 / 3, line_dash="dash", annotation_text="4/3 bound (linear latencies)")
        fig.update_layout(title="Price of Anarchy across demand", xaxis_title="Drivers per hour",
                          yaxis_title="PoA", height=400)
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=ds, y=sw["ne_base"] / ds, name="Selfish, without flyover")
        fig.add_scatter(x=ds, y=sw["ne_fly"] / ds, name="Selfish, with flyover")
        fig.add_scatter(x=ds, y=sw["opt_fly"] / ds, name="Optimum, with flyover", line=dict(dash="dot"))
        braess = sw["ne_fly"] > sw["ne_base"] * (1 + 1e-4)
        if braess.any():
            lo, hi = ds[braess].min(), ds[braess].max()
            fig.add_vrect(x0=lo, x1=hi, fillcolor="orange", opacity=0.15, line_width=0,
                          annotation_text="Braess's paradox", annotation_position="top left")
        fig.update_layout(title="Average trip time", xaxis_title="Drivers per hour",
                          yaxis_title="Minutes per vehicle", height=400)
        st.plotly_chart(fig)
    if braess.any():
        st.info(f"Between about {lo:,.0f} and {hi:,.0f} drivers per hour, opening the flyover makes the "
                "selfish average trip longer. That is Braess's paradox.")
