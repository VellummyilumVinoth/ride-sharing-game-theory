"""Game-Theoretic Simulation Dashboard: Urban Ride-Sharing Ecosystem (CS5824, Group I).

Run with:  streamlit run app.py
"""
import streamlit as st

from ui import common, phase2, tab_1a, tab_1b, tab_1c, tab_2a, tab_2b, tab_overview

st.set_page_config(page_title="Ride-Sharing Game Theory Dashboard", page_icon="🚕", layout="wide")
st.title("Urban Ride-Sharing Ecosystem: Game-Theoretic Simulation Dashboard")
st.caption("Game Theory (CS5824), Group I. Colombo ride-sharing network with competing fleets, riders and a platform.")

params = common.sidebar()

tabs = st.tabs([
    "Overview",
    "1A Normal form & IESDS",
    "1B Extensive form & SPNE",
    "1C Congestion & PoA",
    "2A Repeated game geometry",
    "2B Trigger strategies",
    "Phase 2 (Modules 3 to 6)",
])
renderers = [tab_overview, tab_1a, tab_1b, tab_1c, tab_2a, tab_2b, phase2]
for tab, module in zip(tabs, renderers):
    with tab:
        module.render(params)
