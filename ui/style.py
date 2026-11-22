"""Design system for the dashboard: colours, Plotly template, CSS and page components.

Every tab uses these helpers so the app looks and behaves the same everywhere.
"""
from __future__ import annotations

from contextlib import contextmanager

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------- tokens

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
SURFACE = "#ffffff"
PAGE = "#f9f9f7"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

# One-hue sequential ramp (light -> dark) and the blue <-> red diverging scale with a grey midpoint.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_SCALE = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]
DIV_SCALE = [[0.0, "#e34948"], [0.25, "#f0a3a2"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def _register_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK_2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98, yanchor="top", pad=dict(l=4)),
        colorway=SERIES,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=56, r=24, t=56, b=48),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK_2, size=12), title_text=""),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, zeroline=False, ticks="outside",
                   tickcolor=AXIS, title=dict(font=dict(color=INK_2, size=12)), tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, zeroline=False,
                   title=dict(font=dict(color=INK_2, size=12)), tickfont=dict(color=MUTED)),
        colorscale=dict(sequential=SEQ_SCALE, diverging=DIV_SCALE),
    )
    pio.templates["ridegt"] = t
    pio.templates.default = "plotly_white+ridegt"


_register_template()

# ---------------------------------------------------------------- CSS

_CSS = f"""
<style>
:root {{ --brand: {BLUE}; --ink: {INK}; --ink2: {INK_2}; --muted: {MUTED}; --line: {BORDER}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stDecoration"] {{ display: none; }}

/* hero banner */
.gt-hero {{ background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); color: #fff;
  border-radius: 16px; padding: 26px 30px; margin-bottom: 18px; box-shadow: 0 6px 24px rgba(13,54,107,.18); }}
.gt-hero h1 {{ color: #fff; font-size: 1.85rem; margin: 0 0 6px 0; line-height: 1.2; font-weight: 700; }}
.gt-hero p {{ color: rgba(255,255,255,.88); margin: 0; font-size: 1rem; }}
.gt-hero .gt-tags {{ margin-top: 14px; }}
.gt-hero .gt-tag {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); color: #fff; }}

/* module header */
.gt-mod {{ background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 18px 22px;
  margin: 4px 0 14px 0; box-shadow: 0 1px 2px rgba(11,11,11,.04); }}
.gt-mod .gt-code {{ display: inline-block; background: var(--brand); color: #fff; font-weight: 700;
  border-radius: 8px; padding: 2px 10px; font-size: .85rem; margin-right: 10px; }}
.gt-mod .gt-title {{ font-size: 1.35rem; font-weight: 700; color: var(--ink); vertical-align: middle; }}
.gt-mod .gt-q {{ color: var(--ink2); margin: 10px 0 8px 0; font-size: .98rem; }}
.gt-mod .gt-owner {{ color: var(--muted); font-size: .85rem; float: right; margin-top: 6px; }}
.gt-tag {{ display: inline-block; background: #eef4fc; color: #184f95; border: 1px solid #cde2fb;
  border-radius: 999px; padding: 2px 10px; font-size: .78rem; margin: 2px 6px 2px 0; font-weight: 500; }}

/* section titles inside bordered containers */
.gt-sec {{ font-size: 1.02rem; font-weight: 650; color: var(--ink); margin: 0 0 2px 0; }}
.gt-sub {{ color: var(--muted); font-size: .86rem; margin: 0 0 8px 0; }}

/* bordered containers look like cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 14px !important;
  border-color: var(--line) !important; box-shadow: 0 1px 2px rgba(11,11,11,.04); }}

/* KPI tiles */
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 16px; box-shadow: 0 1px 2px rgba(11,11,11,.04); }}
div[data-testid="stMetricLabel"] p {{ color: var(--ink2); font-size: .82rem; font-weight: 500; }}
div[data-testid="stMetricValue"] {{ color: var(--ink); font-size: 1.55rem; font-weight: 650; }}

/* status badge */
.gt-badge {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 12px;
  font-weight: 600; font-size: .86rem; border: 1px solid; }}
.gt-good {{ color: #006300; background: #ecf7ec; border-color: #bfe5bf; }}
.gt-bad {{ color: #a12828; background: #fbeeee; border-color: #f1c4c4; }}
.gt-warn {{ color: #7a5200; background: #fff6e0; border-color: #f6dc9a; }}

/* tabs */
div[data-testid="stTabs"] button[role="tab"] {{ padding: 8px 14px; border-radius: 10px 10px 0 0; }}
div[data-testid="stTabs"] button[role="tab"] p {{ font-size: .93rem; font-weight: 600; }}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ background: #fff; }}

/* sidebar */
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
.gt-brand {{ display: flex; align-items: center; gap: 10px; margin: 0 0 6px 0; }}
.gt-brand .gt-logo {{ width: 34px; height: 34px; border-radius: 9px; background: var(--brand); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: .95rem; }}
.gt-brand .gt-name {{ font-weight: 750; color: var(--ink); line-height: 1.1; }}
.gt-brand .gt-sub2 {{ color: var(--muted); font-size: .78rem; }}
.gt-side {{ text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; color: var(--muted);
  font-weight: 700; margin: 14px 0 2px 0; }}

/* module cards on the overview */
.gt-card {{ background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; height: 100%;
  box-shadow: 0 1px 2px rgba(11,11,11,.04); }}
.gt-card .gt-code {{ display: inline-block; background: #eef4fc; color: #184f95; font-weight: 700;
  border-radius: 8px; padding: 1px 8px; font-size: .8rem; }}
.gt-card h4 {{ margin: 8px 0 4px 0; font-size: 1.02rem; color: var(--ink); }}
.gt-card p {{ color: var(--ink2); font-size: .88rem; margin: 0 0 8px 0; }}
.gt-card .gt-who {{ color: var(--muted); font-size: .8rem; }}
.gt-foot {{ color: var(--muted); font-size: .8rem; text-align: center; margin-top: 28px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- components


def hero(title: str, subtitle: str, tags: list[str]) -> None:
    chips = "".join(f'<span class="gt-tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="gt-hero"><h1>{title}</h1><p>{subtitle}</p><div class="gt-tags">{chips}</div></div>',
                unsafe_allow_html=True)


def module_header(code: str, title: str, question: str, concepts: list[str], owner: str) -> None:
    chips = "".join(f'<span class="gt-tag">{c}</span>' for c in concepts)
    st.markdown(
        f'<div class="gt-mod"><span class="gt-owner">Owner · {owner}</span>'
        f'<span class="gt-code">{code}</span><span class="gt-title">{title}</span>'
        f'<div class="gt-q">{question}</div>{chips}</div>',
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str | None = None, subtitle: str | None = None):
    """A bordered card with an optional title and one-line explanation."""
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="gt-sec">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="gt-sub">{subtitle}</div>', unsafe_allow_html=True)
        yield


def badge(ok: bool | None, text: str) -> str:
    """Status pill with an icon and a label, so meaning never rests on colour alone."""
    if ok is None:
        return f'<span class="gt-badge gt-warn">⚠ {text}</span>'
    return f'<span class="gt-badge {"gt-good" if ok else "gt-bad"}">{"✓" if ok else "✕"} {text}</span>'


def show_badge(ok: bool | None, text: str) -> None:
    st.markdown(badge(ok, text), unsafe_allow_html=True)


def how_to_read(text: str) -> None:
    with st.expander("How to read this", icon=":material/help:"):
        st.markdown(text)


def chart(fig: go.Figure, height: int = 420, key: str | None = None) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def footer() -> None:
    st.markdown('<div class="gt-foot">Game Theory (CS5824) · Group I · Urban ride-sharing ecosystem simulator</div>',
                unsafe_allow_html=True)
