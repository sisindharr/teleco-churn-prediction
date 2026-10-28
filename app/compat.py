"""Small compatibility helpers so the app works across Streamlit versions.

Newer Streamlit uses  width="stretch"  where older versions used
use_container_width=True. These wrappers try the new form first.
"""
import streamlit as st


def stretch(fn, *args, **kwargs):
    """Call a Streamlit widget so that it fills the available width."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def plotly(fig, key=None):
    """Render a Plotly figure with our own styling (theme=None keeps our colours)."""
    return stretch(st.plotly_chart, fig, key=key, theme=None,
                   config={"displaylogo": False, "displayModeBar": False})
