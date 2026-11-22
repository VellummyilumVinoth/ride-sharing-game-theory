import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gt import m1a_normal_form as m1a
from gt.params import DomainParams
from ui.common import mix_text
from ui.style import (AQUA, BLUE, DIV_SCALE, INK, MUTED, ORANGE, SERIES, card, chart, how_to_read,
                      module_header)


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


def _styled_matrix(g, rows, cols, pure):
    labels = g.labels
    df = pd.DataFrame([[f"{g.A[i, j]:,.0f}  ·  {g.B[i, j]:,.0f}" for j in range(len(labels))]
                       for i in range(len(labels))],
                      index=[f"F1 {l}" for l in labels], columns=[f"F2 {l}" for l in labels])

    def colour(_):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if (i, j) in pure:
                    out.iloc[i, j] = "background-color:#d9f2e8; font-weight:700; color:#0b0b0b"
                elif i not in rows or j not in cols:
                    out.iloc[i, j] = "color:#b5b3ac; background-color:#f6f5f2"
        return out

    return df.style.apply(colour, axis=None)


def render(p: DomainParams) -> None:
    module_header("1A", "Fare game: normal form and IESDS",
                  "Two fleets set their fares at the same time. Which fares would no rational fleet charge, and "
                  "where does competition settle?",
                  ["Strict dominance", "IESDS", "Nash equilibrium", "Support enumeration", "Zero-sum LP"],
                  "Kulathunga W. P. M. W.")
    m = st.segmented_control("Fare levels per fleet", [3, 4, 5, 6], default=3, key="1a_m",
                             format_func=lambda k: f"{k} × {k}") or 3
    g, rows, cols, steps, eqs, checked = _solve(p, m)
    pure = m1a.pure_nash(g.A, g.B)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fares removed by IESDS", f"{len(steps)}")
    k2.metric("Pure Nash equilibria", f"{sum(e.is_pure for e in eqs)}")
    k3.metric("Mixed Nash equilibria", f"{sum(not e.is_pure for e in eqs)}")
    k4.metric("Support pairs checked", f"{checked:,}")

    with card("Payoff matrix", "Profit per hour for (F1 · F2). Green cells are pure Nash equilibria; "
                               "grey cells involve a fare removed by IESDS."):
        st.dataframe(_styled_matrix(g, rows, cols, pure), width="stretch")
        st.caption("Fares after surge: " + " · ".join(f"{l} Rs {f:,.0f}" for l, f in zip(g.labels, g.fares)))

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with card("IESDS steps", "A fare is removed when another fare earns more against every remaining rival fare."):
            if not steps:
                st.info("No fare is strictly dominated, so IESDS leaves the game unchanged.")
            for k, s in enumerate(steps, 1):
                st.markdown(f"**{k}.** F{s.player} drops **{g.labels[s.removed]}**, beaten by "
                            f"{g.labels[s.dominated_by]}")
            st.markdown(f"**Survivors** · F1: {', '.join(g.labels[r] for r in rows)} · "
                        f"F2: {', '.join(g.labels[c] for c in cols)}")
    with c2:
        with card("Nash equilibria", "Found by support enumeration on the full game."):
            st.dataframe(pd.DataFrame([
                {"Type": "Pure" if e.is_pure else "Mixed", "F1 plays": mix_text(e.p, g.labels),
                 "F2 plays": mix_text(e.q, g.labels), "F1 profit": f"Rs {e.u1:,.0f}", "F2 profit": f"Rs {e.u2:,.0f}"}
                for e in eqs]), hide_index=True, width="stretch")

    with card("Expected-payoff envelope", "F1 randomises over the chosen fares; each line is F2's expected profit "
                                          "from one reply."):
        picks = st.multiselect("F1 mixes over (2 fares for a 2D plot, 3 for 3D)", g.labels, default=g.labels[:2],
                               key="1a_env", max_selections=3)
        idx = [g.labels.index(x) for x in picks]
        if len(idx) == 2:
            pgrid, vals = m1a.envelope_2d(g.B, tuple(idx))
            fig = go.Figure()
            for r in range(vals.shape[1]):
                fig.add_scatter(x=pgrid, y=vals[:, r], mode="lines", name=f"F2 replies {g.labels[r]}",
                                line=dict(width=2, color=SERIES[r % len(SERIES)]),
                                hovertemplate="P = %{x:.2f}<br>Rs %{y:,.0f}<extra>%{fullData.name}</extra>")
            fig.add_scatter(x=pgrid, y=vals.max(axis=1), mode="lines", name="Best reply (upper envelope)",
                            line=dict(width=3, color=INK, dash="dot"), hoverinfo="skip")
            fig.update_layout(xaxis_title=f"Probability F1 charges {picks[0]} (otherwise {picks[1]})",
                              yaxis_title="F2 expected profit (Rs/h)", yaxis_tickformat=",")
            chart(fig, 420, key="1a_env2")
        elif len(idx) == 3:
            P1, P2, surfaces = m1a.envelope_3d(g.B, tuple(idx))
            fig = go.Figure()
            for r, Zr in enumerate(surfaces):
                col = SERIES[r % len(SERIES)]
                fig.add_surface(x=P1, y=P2, z=Zr, name=f"F2 replies {g.labels[r]}", showscale=False, opacity=0.8,
                                colorscale=[[0, col], [1, col]], showlegend=True)
            fig.update_layout(scene=dict(xaxis_title=f"P({picks[0]})", yaxis_title=f"P({picks[1]})",
                                         zaxis_title="F2 profit"), margin=dict(l=0, r=0, t=40, b=0))
            chart(fig, 560, key="1a_env3")
        else:
            st.info("Pick two or three fares.")
        how_to_read("The black dotted line is F2's best reply. Where two coloured lines cross on that envelope, "
                    "F2 is indifferent between those replies, which is exactly the condition support enumeration "
                    "solves for F1's equilibrium mix.")

    with card("Zero-sum zone game", "Each fleet parks idle cars in one pickup zone. F1's payoff is its market-share "
                                    "lead, so F2's payoff is the negative. Solved by linear programming."):
        z1, z2, z3, z4 = st.columns(4)
        n = z1.slider("Zones", 3, 10, 7, key="1a_zones")
        r1 = z2.slider("F1 reach (zones)", 0, 4, 1, key="1a_r1")
        r2 = z3.slider("F2 reach (zones)", 0, 4, 2, key="1a_r2")
        seed = z4.number_input("Demand pattern", 0, 999, 7, key="1a_seed")
        zg, (pz, qz, v) = _zone(p.Q, n, r1, r2, int(seed))
        k1, k2, k3 = st.columns(3)
        k1.metric("Value of the game", f"{v:+.1%}", help="F1's guaranteed market-share lead under optimal play.")
        k2.metric("Zones F1 uses", f"{int((pz > 1e-6).sum())}")
        k3.metric("Zones F2 uses", f"{int((qz > 1e-6).sum())}")
        c1, c2 = st.columns(2)
        with c1:
            zmax = float(np.abs(zg.Z).max()) or 1.0
            fig = go.Figure(go.Heatmap(z=zg.Z, x=zg.zones, y=zg.zones, colorscale=DIV_SCALE, zmin=-zmax, zmax=zmax,
                                       xgap=2, ygap=2, colorbar=dict(title="F1 lead", tickformat=".0%"),
                                       hovertemplate="F1 %{y}, F2 %{x}<br>F1 lead %{z:.1%}<extra></extra>"))
            fig.update_layout(title="Payoff to F1 (blue = F1 ahead)", xaxis_title="F2 zone", yaxis_title="F1 zone")
            chart(fig, 420, key="1a_zheat")
        with c2:
            fig = go.Figure()
            fig.add_bar(x=zg.zones, y=pz, name="F1 optimal mix", marker_color=BLUE)
            fig.add_bar(x=zg.zones, y=qz, name="F2 optimal mix", marker_color=ORANGE)
            fig.add_scatter(x=zg.zones, y=zg.demand / zg.demand.sum(), name="Share of demand", mode="lines+markers",
                            line=dict(color=AQUA, width=2), marker=dict(size=8))
            fig.update_layout(title="Optimal mixed strategies", barmode="group", bargap=0.3, bargroupgap=0.08,
                              yaxis_title="Probability", yaxis_tickformat=".0%")
            fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
            chart(fig, 420, key="1a_zbar")
        st.latex(r"\max_{p,\,w}\ w\quad\text{s.t.}\quad \sum_i p_i Z_{ij}\ge w\ \ \forall j,\quad \sum_i p_i=1,\ p\ge 0")
