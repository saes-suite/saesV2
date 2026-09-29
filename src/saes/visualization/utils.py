"""Visualization utility functions for palette mapping and plot formatting."""

import re
from typing import Any

import plotly.express as px

# Available palette names mapped to Plotly color lists
PALETTE_MAP: dict[str, list[str]] = {
    "Plotly": px.colors.qualitative.Plotly,
    "Set1": px.colors.qualitative.Set1,
    "Set2": px.colors.qualitative.Set2,
    "Pastel": px.colors.qualitative.Pastel,
    "Dark24": px.colors.qualitative.Dark24,
    "Bold": px.colors.qualitative.Bold,
    "Safe": px.colors.qualitative.Safe,
    "Vivid": px.colors.qualitative.Vivid,
    "Viridis": px.colors.sequential.Viridis,
    "Plasma": px.colors.sequential.Plasma,
    "Turbo": px.colors.sequential.Turbo,
}


def to_hex_color(color_str: str) -> str:
    """Convert any color string (hex, rgb(r,g,b), rgba(r,g,b,a)) to valid 7-char #RRGGBB hex string.

    Args:
        color_str: Raw color specification string.

    Returns:
        str: Standard uppercase #RRGGBB hex string.
    """
    s = color_str.strip()
    if s.startswith("#"):
        if len(s) == 4:
            return f"#{s[1] * 2}{s[2] * 2}{s[3] * 2}".upper()
        if len(s) == 7:
            return s.upper()

    rgb_match = re.match(r"^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", s, re.IGNORECASE)
    if rgb_match:
        r, g, b = (int(rgb_match.group(i)) for i in range(1, 4))
        r_c = max(0, min(255, r))
        g_c = max(0, min(255, g))
        b_c = max(0, min(255, b))
        return f"#{r_c:02x}{g_c:02x}{b_c:02x}".upper()

    return "#1F77B4"


def get_color_map(
    algorithms: list[str],
    palette_name: str = "Plotly",
    custom_colors: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a mapping from algorithm name to standard hex color string.

    Args:
        algorithms: List of unique algorithm names.
        palette_name: Predefined palette identifier.
        custom_colors: Optional user-specified per-algorithm color overrides.

    Returns:
        dict[str, str]: Map of algorithm -> hex color string.
    """
    colors = PALETTE_MAP.get(palette_name, px.colors.qualitative.Plotly)
    color_map: dict[str, str] = {}
    for i, alg in enumerate(algorithms):
        if custom_colors and alg in custom_colors:
            color_map[alg] = to_hex_color(custom_colors[alg])
        else:
            raw_color = colors[i % len(colors)]
            color_map[alg] = to_hex_color(raw_color)
    return color_map


def format_plot_title(kwargs: Any, default_suffix: str = "") -> str:
    """Format plot title based on kwargs parameters."""
    if title := kwargs.get("title"):
        return str(title)

    instance = kwargs.get("instance", "")
    metric = kwargs.get("metric", "value")

    if instance and metric:
        return f"{default_suffix} ({instance} — {metric})"
    elif metric:
        return f"{default_suffix} ({metric})"
    return default_suffix
