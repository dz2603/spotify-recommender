"""Reusable Streamlit UI components."""

from collections.abc import Iterable
from contextlib import contextmanager

import streamlit as st

from src.ui import theme
from src.ui.common import spotify_player


def page_header(title: str, description: str | None = None):
    st.header(title)
    if description:
        st.markdown(description)


def divider():
    st.markdown("---")


@contextmanager
def control_panel(title: str, description: str | None = None):
    """Group page controls in a consistent bordered panel."""
    st.markdown(f"#### {title}")
    with st.container(border=True):
        if description:
            st.caption(description)
        yield


def info_note(body: str):
    st.info(body)


def render_metric_cards(metrics: Iterable[tuple[str, str]]):
    metrics = list(metrics)
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def format_artists(artists) -> str:
    return ", ".join(artists) if isinstance(artists, list) else str(artists)


def render_reference_track(row, height: int = 152):
    artists = format_artists(row["artists"])
    st.success(f"🎧 Reference: **{row['track_name']}** by *{artists}*")
    spotify_player(row["track_id"], height=height)


def render_spotify_grid(data, results, columns: int = 2, height: int = 80):
    st.markdown("### ⏯️ Listen to Recommendations")
    cols = st.columns(columns)
    for i, result in enumerate(results):
        with cols[i % columns]:
            row = data.iloc[result["index"]]
            st.caption(f"#{i + 1}: {row['track_name']} — {format_artists(row['artists'])}")
            spotify_player(row["track_id"], height=height)


def themed_card(markdown: str):
    st.markdown(
        f"""
        <div style="{theme.card_style()}">
            {markdown}
        </div>
        """,
        unsafe_allow_html=True,
    )
