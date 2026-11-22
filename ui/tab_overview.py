import pandas as pd
import streamlit as st

from gt.params import DomainParams
from ui.common import TEAM


def render(p: DomainParams) -> None:
    st.subheader("Ecosystem")
    st.markdown(
        "We model a ride-sharing system in Colombo with three kinds of agents. **Competing fleets** "
        "set fares and choose zones and routes. **Riders** arrive at rate Q per hour and move to the "
        "cheaper fleet according to the price sensitivity β. **The platform** sets the surge multiplier "
        "σ, the commission κ and driver bonuses b, and in Phase 2 it acts as the regulator. Every "
        "payoff in this dashboard is generated from the sidebar sliders, so changing one slider "
        "updates every module."
    )
    st.markdown("**Fare payoff used by 1A, 2A and 2B**")
    st.latex(r"D_i=\min\{Q,\ \max\{0,\ Q/2-\beta(a_i-a_j)\}\},\qquad "
             r"u_i(a)=\big((1-\kappa)a_i-c\big)\min\{D_i(a),K\}")

    st.subheader("Current parameters")
    st.dataframe(pd.DataFrame([
        ("Commuter demand Q", f"{p.Q:,.0f} riders/hour"),
        ("Surge multiplier σ", f"{p.sigma:.1f}"),
        ("Fare levels (10 km)", "Low Rs 600, Standard Rs 750, Surge Rs 1,000 (× σ)"),
        ("Operating cost c", f"Rs {p.c:,.0f} per trip"),
        ("Commission κ", f"{p.kappa:.0%}"),
        ("Price sensitivity β", f"{p.beta:.1f} riders per Rs"),
        ("Fleet capacity K", f"{p.K:,.0f} rides/hour"),
        ("Road capacity scale", f"{p.cap_scale:.1f}×"),
        ("Patience δ", f"{p.delta:.2f}"),
    ], columns=["Parameter", "Value"]), hide_index=True)

    st.subheader("Team and module owners")
    st.dataframe(pd.DataFrame(TEAM, columns=["Name", "Phase 1", "Phase 2"]), hide_index=True)
