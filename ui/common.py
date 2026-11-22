"""Shared sidebar and small display helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gt.params import DomainParams

TEAM = [
    ("Kulathunga W. P. M. W.", "Module 1A (Normal Form and IESDS)", "Cross-module integration and report (Part A)"),
    ("Vishmani P. C.", "Module 1B (Extensive Form and SPNE)", "Cross-module integration and report (Part B)"),
    ("Wijerathne R. M. N. S.", "Module 1C (Congestion and Price of Anarchy)", "Module 3 (Learning Dynamics)"),
    ("Karunarathna W. K.", "Module 2A (Repeated Game Geometry and UI)", "Module 4 (Bayesian Games and Central Control)"),
    ("Vinoth V.", "Module 2B (Trigger Strategies and UI)", "Module 5 (Coalitional Games and the Shapley Value)"),
    ("Wijesekera H. T. J.", "Dashboard architecture (Phase 1 support)", "Module 6 (Adversarial Stress-Testing and RL Collusion)"),
]


# key: (label, min, max, default, step, help)
SLIDERS = {
    "Q": ("Commuter demand Q (riders/hour)", 100, 5000, 400, 50, "Riders who request a trip in the zone each hour."),
    "beta": ("Price sensitivity β (riders per Rs)", 0.1, 3.0, 1.0, 0.1,
             "How many riders switch to the cheaper fleet for each rupee of fare gap."),
    "sigma": ("Surge multiplier σ", 1.0, 3.0, 1.0, 0.1, "Multiplies every fare level."),
    "c": ("Operating cost per trip c (Rs)", 100, 600, 350, 10, "Fuel, wear and driver time for one 10 km trip."),
    "kappa": ("Platform commission κ (%)", 0, 30, 20, 1, "Share of each fare the platform keeps."),
    "K": ("Fleet capacity K (rides/hour)", 50, 5000, 400, 50, "Most rides one fleet can serve per hour."),
    "cap": ("Road capacity scale kₑ (×)", 0.5, 2.0, 1.0, 0.1, "Multiplies every road's capacity in Module 1C."),
    "delta": ("Patience δ", 0.50, 0.99, 0.90, 0.01,
              "How much fleets value tomorrow's profit (discount factor) in the repeated game."),
}
GROUPS = [("Market", ["Q", "beta"]), ("Pricing", ["sigma", "c", "kappa"]),
          ("Capacity", ["K", "cap"]), ("Repeated game", ["delta"])]


def _reset() -> None:
    for k, spec in SLIDERS.items():
        st.session_state[f"p_{k}"] = spec[3]


def sidebar() -> DomainParams:
    """The shared DomainParams sliders. Every module builds its payoffs from these values."""
    sb = st.sidebar
    sb.markdown('<div class="gt-brand"><div class="gt-logo">GT</div><div><div class="gt-name">Ride-Sharing '
                'Strategy Lab</div><div class="gt-sub2">CS5824 · Group I</div></div></div>', unsafe_allow_html=True)
    sb.caption("Scenario controls are shared by every module. Fares are for a 10 km trip in Colombo.")
    vals = {}
    for group, keys in GROUPS:
        sb.markdown(f'<div class="gt-side">{group}</div>', unsafe_allow_html=True)
        for k in keys:
            label, lo, hi, default, step, help_ = SLIDERS[k]
            vals[k] = sb.slider(label, lo, hi, default, step, key=f"p_{k}", help=help_)
    sb.button("Reset to defaults", on_click=_reset, icon=":material/restart_alt:", width="stretch")
    return DomainParams(Q=float(vals["Q"]), sigma=float(vals["sigma"]), c=float(vals["c"]),
                        kappa=float(vals["kappa"]) / 100, beta=float(vals["beta"]), K=float(vals["K"]),
                        cap_scale=float(vals["cap"]), delta=float(vals["delta"]))


def rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def bimatrix_frame(A: np.ndarray, B: np.ndarray, labels: list[str], highlight=None) -> pd.DataFrame:
    """Payoff cells written as 'F1 , F2' in Rs per hour."""
    cells = [[f"{A[i, j]:,.0f} , {B[i, j]:,.0f}" for j in range(A.shape[1])] for i in range(A.shape[0])]
    return pd.DataFrame(cells, index=[f"F1: {l}" for l in labels], columns=[f"F2: {l}" for l in labels])


def mix_text(x: np.ndarray, labels: list[str]) -> str:
    parts = [f"{l} {p:.2f}" for l, p in zip(labels, x) if p > 1e-6]
    return ", ".join(parts)


def owner(name: str) -> None:
    st.caption(f"Module owner: {name}")
