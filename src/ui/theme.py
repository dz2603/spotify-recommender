"""Centralized UI styling constants and small style helpers."""

PAGE_ICON = "🎵"
APP_TITLE = "Spotify Recommender"
SIDEBAR_CAPTION = "CS Project — KNN · K-Means · GMM · PCA · AE"

PRIMARY = "#1DB954"
BACKGROUND = "#0E1117"
SURFACE = "#161B22"
SURFACE_ALT = "#121212"
BORDER = "#263241"
TEXT = "#F5F5F5"
TEXT_MUTED = "#A9B4C0"

RADIUS_SM = "8px"
RADIUS_MD = "12px"
SPACING_SM = "8px"
SPACING_MD = "12px"
SPACING_LG = "20px"

SPOTIFY_EMBED_RADIUS = RADIUS_MD
SPOTIFY_EMBED_BG = SURFACE_ALT
SPOTIFY_EMBED_MARGIN_BOTTOM = SPACING_MD


def card_style() -> str:
    return (
        f"background:{SURFACE};"
        f"border:1px solid {BORDER};"
        f"border-radius:{RADIUS_MD};"
        f"padding:{SPACING_LG};"
    )
