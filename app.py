"""Game-Theoretic Simulation Dashboard: Urban Ride-Sharing Ecosystem (CS5824, Group I).

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Ride-Sharing Strategy Lab", page_icon=":material/local_taxi:", layout="wide",
                   initial_sidebar_state="expanded")

from ui import common, phase2, style, tab_1a, tab_1b, tab_1c, tab_2a, tab_2b, tab_overview  # noqa: E402

style.inject_css()
params = common.sidebar()

tabs = st.tabs([
    ":material/dashboard: Overview",
    ":material/grid_on: 1A Fare game",
    ":material/account_tree: 1B Surge & entry",
    ":material/traffic: 1C Congestion",
    ":material/hexagon: 2A Repeated geometry",
    ":material/handshake: 2B Collusion",
    ":material/upcoming: Phase 2",
])
renderers = [tab_overview, tab_1a, tab_1b, tab_1c, tab_2a, tab_2b, phase2]
for tab, module in zip(tabs, renderers):
    with tab:
        module.render(params)

style.footer()
