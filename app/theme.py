"""Design system: colour tokens, global CSS (light + dark) and Plotly styling.

Streamlit cannot switch its built-in theme at runtime, so dark/light mode is
implemented with CSS variables that we override on every run.
"""
from __future__ import annotations

FONT = "'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"
ACCENT = "#22D3EE"

RISK_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
CHURN_COLORS = {"Retained": "#6366F1", "Churned": "#F43F5E"}

THEMES = {
    "light": {
        "bg": "#F4F6FB", "surface": "#FFFFFF", "surface2": "#F1F4F9", "text": "#0F172A",
        "muted": "#5B6B82", "border": "#E3E8F0", "primary": "#5B5BD6",
        "primary_soft": "rgba(91,91,214,0.10)", "shadow": "0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05)",
    },
    "dark": {
        "bg": "#0A0F1E", "surface": "#121A2F", "surface2": "#18223B", "text": "#E8ECF6",
        "muted": "#97A3BC", "border": "#243252", "primary": "#8B8BFF",
        "primary_soft": "rgba(139,139,255,0.14)", "shadow": "0 1px 2px rgba(0,0,0,.30), 0 8px 24px rgba(0,0,0,.25)",
    },
}

_STATIC_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, button, input, textarea, select { font-family: var(--font) !important; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
.block-container { max-width: 1280px; padding: 2rem 2rem 4rem; }

/* ---- text colours (Streamlit sets its own, so we override explicitly) ---- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stCheckbox"] label p, [data-testid="stExpander"] summary p { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .2rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .6rem .85rem; border-radius: 10px; width: 100%; cursor: pointer; transition: background .15s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--surface2); }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--primary-soft); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: var(--primary); font-weight: 700; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 500; font-size: .95rem; }

/* ---- inputs ---- */
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"],
[data-testid="stNumberInput"] input, .stTextInput input {
  background-color: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: var(--text); }
div[data-baseweb="select"] svg { fill: var(--muted); }
[data-testid="stNumberInput"] button { background: var(--surface2) !important; color: var(--text) !important; }
ul[role="listbox"], div[data-baseweb="popover"] > div, div[data-baseweb="menu"] {
  background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border);
}
li[role="option"] { color: var(--text) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: var(--primary-soft) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted); }
[data-testid="stSliderThumbValue"] { color: var(--primary); }
[data-baseweb="tag"] { background: var(--primary-soft) !important; }
[data-baseweb="tag"] span { color: var(--text) !important; }

/* ---- forms, tabs, expanders ---- */
[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.4rem; box-shadow: var(--shadow); }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--primary); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--primary); }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }
[data-testid="stExpander"] summary { color: var(--text); }

/* ---- buttons ---- */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, var(--primary), #22D3EE); border: 0; border-radius: 10px; font-weight: 700;
}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: #fff !important; }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; font-weight: 600;
}
[data-testid="stBaseButton-secondary"] p { color: var(--text); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--primary); }

/* ---- brand ---- */
.brand { display: flex; align-items: center; gap: .75rem; padding: .25rem .25rem 1rem; }
.brand-logo { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; color: #fff;
  font-weight: 800; font-size: 1.15rem; background: linear-gradient(135deg, var(--primary), #22D3EE); }
.brand-name { font-weight: 800; font-size: 1.15rem; color: var(--text); line-height: 1.1; }
.brand-tag { font-size: .75rem; color: var(--muted); }
.side-card { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: .8rem .9rem; font-size: .82rem; color: var(--muted); }
.side-card b { color: var(--text); }

/* ---- page header ---- */
.page-head { margin-bottom: 1.4rem; }
.page-title { font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); margin: 0; }
.page-sub { color: var(--muted); font-size: 1rem; margin-top: .25rem; max-width: 70ch; }
.section-title { font-weight: 700; font-size: 1.05rem; color: var(--text); margin: 1.6rem 0 .6rem; }
.section-note { color: var(--muted); font-size: .88rem; margin: -.35rem 0 .8rem; }

/* ---- cards ---- */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.2rem 1.3rem; box-shadow: var(--shadow); margin-bottom: 1rem; }
.card-title { font-weight: 700; color: var(--text); margin-bottom: .4rem; }
.card-text { color: var(--muted); font-size: .93rem; line-height: 1.55; }
.card-text b { color: var(--text); }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: var(--shadow); height: 100%; margin-bottom: .6rem; }
.kpi-head { display: flex; justify-content: space-between; align-items: center; }
.kpi-label { color: var(--muted); font-size: .85rem; font-weight: 600; }
.kpi-icon { width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; font-size: .95rem; background: var(--tone-soft, var(--primary-soft)); }
.kpi-value { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); margin: .35rem 0 .15rem; }
.kpi-foot { font-size: .8rem; color: var(--muted); }

/* ---- badges, risk bar ---- */
.badge { display: inline-block; padding: .22rem .7rem; border-radius: 999px; font-weight: 700; font-size: .8rem; }
.badge-Low { background: rgba(16,185,129,.15); color: #0E9F6E; }
.badge-Medium { background: rgba(245,158,11,.17); color: #C77D05; }
.badge-High { background: rgba(239,68,68,.15); color: #DC2626; }
.risk-bar { position: relative; height: 12px; border-radius: 99px; overflow: hidden; background: linear-gradient(90deg, rgba(16,185,129,.30) 0 30%, rgba(245,158,11,.30) 30% 60%, rgba(239,68,68,.30) 60% 100%); }
.risk-fill { height: 100%; border-radius: 99px; }
.risk-scale { display: flex; justify-content: space-between; font-size: .72rem; color: var(--muted); margin-top: .3rem; }
.verdict { font-size: 1rem; color: var(--text); margin: .8rem 0 .2rem; }

/* ---- recommendations / callouts ---- */
.rec { background: var(--surface2); border: 1px solid var(--border); border-left: 4px solid var(--tone, var(--primary)); border-radius: 12px; padding: .75rem 1rem; margin-bottom: .6rem; }
.rec-title { font-weight: 700; color: var(--text); font-size: .93rem; }
.rec-title span { font-weight: 600; font-size: .72rem; color: var(--tone); margin-left: .5rem; }
.rec-text { color: var(--muted); font-size: .86rem; margin-top: .15rem; line-height: 1.5; }
.callout { background: var(--primary-soft); border-radius: 14px; padding: 1rem 1.2rem; margin: .5rem 0 1rem; }
.callout-title { font-weight: 700; color: var(--text); }
.callout-text { color: var(--muted); font-size: .92rem; margin-top: .2rem; line-height: 1.55; }
.empty { text-align: center; padding: 2.5rem 1rem; color: var(--muted); border: 1.5px dashed var(--border); border-radius: 16px; }

/* ---- tables ---- */
.tbl-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); }
.tbl { width: 100%; border-collapse: collapse; font-size: .88rem; }
.tbl th { text-align: left; padding: .7rem .9rem; color: var(--muted); font-weight: 600; background: var(--surface2); white-space: nowrap; }
.tbl td { padding: .65rem .9rem; color: var(--text); border-top: 1px solid var(--border); white-space: nowrap; }
.tbl tr.best td { background: var(--primary-soft); font-weight: 700; }

/* ---- about page ---- */
.step { display: flex; gap: .9rem; margin-bottom: .9rem; }
.step-n { flex: 0 0 30px; height: 30px; border-radius: 50%; background: var(--primary-soft); color: var(--primary); font-weight: 800; display: grid; place-items: center; font-size: .85rem; }
.chip { display: inline-block; padding: .3rem .8rem; margin: 0 .4rem .5rem 0; border-radius: 999px; background: var(--surface2); border: 1px solid var(--border); color: var(--text); font-size: .85rem; font-weight: 600; }

/* ---- responsive ---- */
@media (max-width: 768px) {
  .block-container { padding: 1rem .9rem 3rem; }
  .page-title { font-size: 1.45rem; }
  .kpi { padding: .9rem 1rem; }
  .kpi-value { font-size: 1.5rem; }
  [data-testid="stForm"] { padding: 1rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def css(dark: bool) -> str:
    """Return the full <style> block for the chosen mode."""
    t = THEMES["dark" if dark else "light"]
    variables = (
        ":root{" + f"--font:{FONT};--bg:{t['bg']};--surface:{t['surface']};--surface2:{t['surface2']};"
        f"--text:{t['text']};--muted:{t['muted']};--border:{t['border']};--primary:{t['primary']};"
        f"--primary-soft:{t['primary_soft']};--shadow:{t['shadow']};" + "}"
    )
    return f"<style>{variables}{_STATIC_CSS}</style>"


def style_fig(fig, dark: bool, height: int = 340, legend: bool = True, title: str | None = None):
    """Apply consistent look & feel to any Plotly figure."""
    t = THEMES["dark" if dark else "light"]
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=t["text"], size=12),
        margin=dict(l=8, r=8, t=48 if title else 16, b=48 if legend else 8),
        height=height, showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, title_text=""),
        colorway=["#6366F1", "#22D3EE", "#F43F5E", "#F59E0B", "#10B981", "#A78BFA"],
        hoverlabel=dict(font_family=FONT),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=t["border"], linecolor=t["border"], zeroline=False)
    fig.update_yaxes(gridcolor=t["border"], linecolor=t["border"], zeroline=False)
    return fig
