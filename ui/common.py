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


def sidebar() -> DomainParams:
    """The shared DomainParams sliders. Every module builds its payoffs from these values."""
    st.sidebar.header("Domain parameters")
    st.sidebar.caption("Shared by every module. Fares are for a 10 km trip in Colombo.")
    Q = st.sidebar.slider("Commuter demand Q (riders/hour)", 100, 5000, 400, 50)
    sigma = st.sidebar.slider("Surge multiplier σ", 1.0, 3.0, 1.0, 0.1)
    c = st.sidebar.slider("Operating cost per trip c (Rs)", 100, 600, 350, 10)
    kappa = st.sidebar.slider("Platform commission κ (%)", 0, 30, 20, 1) / 100
    beta = st.sidebar.slider("Rider price sensitivity β (riders per Rs)", 0.1, 3.0, 1.0, 0.1)
    K = st.sidebar.slider("Fleet capacity K (rides/hour)", 50, 5000, 400, 50)
    cap = st.sidebar.slider("Road capacity scale kₑ (×)", 0.5, 2.0, 1.0, 0.1)
    delta = st.sidebar.slider("Patience δ", 0.50, 0.99, 0.90, 0.01)
    return DomainParams(Q=float(Q), sigma=float(sigma), c=float(c), kappa=float(kappa),
                        beta=float(beta), K=float(K), cap_scale=float(cap), delta=float(delta))


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
