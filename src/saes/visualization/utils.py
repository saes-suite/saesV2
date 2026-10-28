"""Visualization utility functions for palette mapping and plot formatting."""

import re
from typing import Any

import plotly.express as px

# Available palette names mapped to Plotly color lists (includes scientific palettes matching SAES-UI)
PALETTE_MAP: dict[str, list[str]] = {
    "academic": ["#0284C7", "#6366F1", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"],
    "nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"],
    "viridis": ["#440154", "#3B528B", "#21908D", "#5DC963", "#FDE725"],
    "colorblind": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
    "set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"],
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

# Standard Scientific Paper Presets (width_px, height_px) matching SAES-UI
PAPER_FORMAT_PRESETS: dict[str, tuple[int, int]] = {
    "single-col": (680, 510),
    "golden": (800, 495),
    "double-col": (960, 540),
    "square": (600, 600),
    "compact": (920, 440),
}


def resolve_figure_dimensions(kwargs: Any, default_width: int = 800, default_height: int = 495) -> tuple[int, int]:
    """Resolve (width, height) in pixels from paper_format preset or explicit width/height kwargs."""
    paper_format = kwargs.get("paper_format")
    if isinstance(paper_format, str) and paper_format in PAPER_FORMAT_PRESETS:
        w_preset, h_preset = PAPER_FORMAT_PRESETS[paper_format]
    else:
        w_preset, h_preset = default_width, default_height

    width = int(kwargs.get("width", w_preset))
    height = int(kwargs.get("height", h_preset))
    return width, height


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
