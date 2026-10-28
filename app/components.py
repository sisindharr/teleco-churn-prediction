"""Reusable UI building blocks (HTML cards, badges, tables, gauge)."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from src.config import HIGH_MIN, LOW_MAX
from src.predict import risk_band


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def esc(value) -> str:
    """HTML-escape text. '$' is escaped too so Streamlit does not read it as LaTeX."""
    return html.escape(str(value)).replace("$", "&#36;")


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"


def goto(page: str) -> None:
    """Button callback: jump to another page via the sidebar radio."""
    st.session_state["nav"] = page


# ----------------------------------------------------------------------------
# Layout pieces
# ----------------------------------------------------------------------------
def page_header(title: str, subtitle: str) -> None:
    md(f'<div class="page-head"><h1 class="page-title">{esc(title)}</h1>'
       f'<div class="page-sub">{esc(subtitle)}</div></div>')


def section(title: str, note: str = "") -> None:
    md(f'<div class="section-title">{esc(title)}</div>')
    if note:
        md(f'<div class="section-note">{esc(note)}</div>')


def kpi_card(label: str, value: str, foot: str = "", icon: str = "", tone: str = "#6366F1") -> None:
    soft = tone + "26"  # 15% opacity hex
    md(f'<div class="kpi" style="--tone-soft:{soft}"><div class="kpi-head">'
       f'<span class="kpi-label">{esc(label)}</span><span class="kpi-icon">{icon}</span></div>'
       f'<div class="kpi-value">{esc(value)}</div><div class="kpi-foot">{esc(foot)}</div></div>')


def card(title: str, body: str) -> None:
    """Card with a title and trusted HTML body (body must already be escaped)."""
    md(f'<div class="card"><div class="card-title">{esc(title)}</div><div class="card-text">{body}</div></div>')


def callout(title: str, body: str) -> None:
    md(f'<div class="callout"><div class="callout-title">{esc(title)}</div>'
       f'<div class="callout-text">{esc(body)}</div></div>')


def empty_state(text: str) -> None:
    md(f'<div class="empty">{esc(text)}</div>')


def risk_badge(level: str) -> str:
    return f'<span class="badge badge-{level}">{level} risk</span>'


def risk_bar(prob: float) -> str:
    color = theme.RISK_COLORS[risk_band(prob)]
    return (f'<div class="risk-bar"><div class="risk-fill" style="width:{max(prob * 100, 2):.1f}%;background:{color}"></div></div>'
            f'<div class="risk-scale"><span>0%</span><span>{int(LOW_MAX * 100)}%</span><span>{int(HIGH_MIN * 100)}%</span><span>100%</span></div>')


def recommendation_cards(recs: list[dict]) -> None:
    tone = {"High": theme.RISK_COLORS["High"], "Medium": theme.RISK_COLORS["Medium"], "Low": theme.RISK_COLORS["Low"]}
    for r in recs:
        md(f'<div class="rec" style="--tone:{tone[r["priority"]]}"><div class="rec-title">{esc(r["title"])}'
           f'<span>{esc(r["priority"])} priority</span></div><div class="rec-text">{esc(r["detail"])}</div></div>')


def html_table(df: pd.DataFrame, risk_col: str | None = None, highlight_row: int | None = None) -> None:
    """Theme-aware table (st.dataframe cannot follow our dark/light switch)."""
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col, val in row.items():
            cells.append(f"<td>{risk_badge(val) if col == risk_col else esc(val)}</td>")
        cls = ' class="best"' if highlight_row == i else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    st.html(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def risk_gauge(prob: float, dark: bool) -> go.Figure:
    t = theme.THEMES["dark" if dark else "light"]
    color = theme.RISK_COLORS[risk_band(prob)]
    low, high = round(LOW_MAX * 100), round(HIGH_MIN * 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 46, "color": t["text"], "family": theme.FONT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": t["muted"], "tickfont": {"size": 10, "color": t["muted"]}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, low], "color": "rgba(16,185,129,.22)"},
                {"range": [low, high], "color": "rgba(245,158,11,.22)"},
                {"range": [high, 100], "color": "rgba(239,68,68,.22)"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=24, r=24, t=16, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=theme.FONT, color=t["text"]))
    return fig
