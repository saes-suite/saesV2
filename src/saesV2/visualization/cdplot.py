"""Critical Distance (CD) diagram generator (Demšar 2006) for multi-algorithm comparisons."""

from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from saesV2.domain.exceptions import VisualizationError
from saesV2.domain.models import CriticalDistanceResult
from saesV2.stats.ranking import FriedmanNemenyiTest
from saesV2.visualization.base import Plotter
from saesV2.visualization.utils import get_color_map, resolve_figure_dimensions


class CriticalDistancePlotGenerator(Plotter):
    """Generates Demšar (2006) Friedman + Nemenyi Critical Distance (CD) diagrams."""

    def __init__(self, palette: str = "academic") -> None:
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Critical Distance (CD) Diagram"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return (
            "Plots Friedman average ranks across algorithms with Nemenyi post-hoc "
            "Critical Distance (CD) cliques connecting statistically indistinguishable methods."
        )

    def _resolve_cd_result(self, df: pd.DataFrame, **kwargs: Any) -> CriticalDistanceResult:
        """Obtain CriticalDistanceResult from kwargs or compute it from DataFrame."""
        existing = kwargs.get("cd_result")
        if isinstance(existing, CriticalDistanceResult):
            return existing

        required_cols = {"algorithm", "value"}
        if not required_cols.issubset(df.columns):
            raise VisualizationError(
                "DataFrame must contain 'algorithm' and 'value' columns for Critical Distance plot."
            )

        work_df = df.copy()
        metric = str(kwargs.get("metric") or (work_df["metric"].iloc[0] if "metric" in work_df.columns else "value"))
        if "metric" not in work_df.columns:
            work_df["metric"] = metric
        if "instance" not in work_df.columns:
            work_df["instance"] = str(kwargs.get("instance") or "Benchmark")
        if "run_id" not in work_df.columns:
            work_df["run_id"] = work_df.groupby(["algorithm", "instance"]).cumcount()
        if "maximize" not in work_df.columns:
            work_df["maximize"] = bool(kwargs.get("maximize", True))

        alpha = float(kwargs.get("alpha", 0.05))
        scope = str(kwargs.get("scope", "all-instances"))
        instance = kwargs.get("instance")

        try:
            return FriedmanNemenyiTest().run(
                df=work_df,
                metric=metric,
                alpha=alpha,
                scope=scope,
                instance=str(instance) if instance else None,
            )
        except Exception as e:
            raise VisualizationError(f"Failed to compute Critical Distance ranks: {e!s}") from e

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a publication-grade Matplotlib Critical Distance (CD) diagram."""
        cd_res = self._resolve_cd_result(df, **kwargs)
        width_px, height_px = resolve_figure_dimensions(kwargs, default_width=800, default_height=460)
        figsize = kwargs.get("figsize", (width_px / 100.0, height_px / 100.0))

        ranks = cd_res.ranks
        k = cd_res.k
        cd = cd_res.cd
        cliques = cd_res.cliques
        palette_name = str(kwargs.get("palette", self.palette))
        color_map = get_color_map([r.algorithm for r in ranks], palette_name, kwargs.get("custom_colors"))

        try:
            fig, ax = plt.subplots(figsize=figsize, dpi=150)

            max_chars = max((len(r.algorithm) for r in ranks), default=8)
            pad_units = max(1.4, (max_chars + 8) * 0.14)
            leader_gap = max(0.25, pad_units * 0.2)
            left_edge = 1.0 - leader_gap
            right_edge = float(k) + leader_gap

            # Main axis line at y = 0
            ax.plot([1, k], [0, 0], color="#0f172a", linewidth=2.0, zorder=3)

            # Ticks along rank axis
            t = 1.0
            while t <= k + 1e-9:
                is_major = abs(t - round(t)) < 1e-9
                tick_h = 0.055 if is_major else 0.03
                ax.plot([t, t], [0, tick_h], color="#0f172a", linewidth=1.5 if is_major else 0.9)
                if is_major:
                    ax.text(
                        t,
                        0.075,
                        f"{int(round(t))}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        color="#0f172a",
                    )
                t += 0.5

            ax.text(
                1.0,
                0.165,
                "← Better (Rank 1)",
                ha="left",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#334155",
            )

            # CD reference bar on top tier above "← Better (Rank 1)"
            cd_span = min(cd, float(k - 1))
            cd_start = max(1.0, float(k) - cd_span)
            cd_end = float(k)
            y_cd = 0.29
            ax.plot([cd_start, cd_end], [y_cd, y_cd], color="#ef4444", linewidth=2.5)
            ax.plot([cd_start, cd_start], [y_cd - 0.025, y_cd + 0.025], color="#ef4444", linewidth=1.8)
            ax.plot([cd_end, cd_end], [y_cd - 0.025, y_cd + 0.025], color="#ef4444", linewidth=1.8)
            ax.text(
                (cd_start + cd_end) / 2.0,
                y_cd + 0.035,
                f"CD = {cd:.2f} (α = {cd_res.alpha})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#ef4444",
            )

            # Cliques
            for c_idx, (r_start, r_end) in enumerate(cliques):
                y_clique = -0.08 - c_idx * 0.055
                ax.plot(
                    [r_start - 0.03, r_end + 0.03],
                    [y_clique, y_clique],
                    color="#0f172a",
                    linewidth=4.5,
                    solid_capstyle="round",
                )

            # L-shaped leader arms and labels
            n_left = math.ceil(len(ranks) / 2)
            max_row_idx = max(0, n_left - 1)
            y_first = -0.15 - len(cliques) * 0.06
            y_step = max(0.14, 0.36 / max_row_idx) if max_row_idx > 0 else 0.15
            for idx, item in enumerate(ranks):
                color = color_map[item.algorithm]
                is_left = idx < n_left
                row_idx = idx if is_left else (len(ranks) - 1 - idx)
                y_leader = y_first - row_idx * y_step
                x_end = left_edge if is_left else right_edge

                ax.plot([item.avg_rank, item.avg_rank], [0, y_leader], color=color, linestyle=":", linewidth=1.4)
                ax.plot([item.avg_rank, x_end], [y_leader, y_leader], color=color, linewidth=1.5)
                ax.scatter([item.avg_rank], [0], color=color, s=45, zorder=5, edgecolors="white", linewidths=1.0)

                label_txt = (
                    f"{item.algorithm} ({item.avg_rank:.2f})" if is_left else f"({item.avg_rank:.2f}) {item.algorithm}"
                )
                ax.text(
                    x_end - 0.06 if is_left else x_end + 0.06,
                    y_leader,
                    label_txt,
                    ha="right" if is_left else "left",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#0f172a",
                )

            min_y = (y_first - max_row_idx * y_step) - 0.055
            ax.set_xlim(1.0 - pad_units, float(k) + pad_units)
            ax.set_ylim(min_y, 0.36)
            ax.axis("off")

            default_title = f"{cd_res.metrics_label}  •  {cd_res.instances_label}\n{cd_res.params_label}"
            title = kwargs.get("title", default_title)
            ax.set_title(title, fontsize=10, fontweight="bold", pad=12)

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate Matplotlib CD diagram: {e!s}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly Critical Distance (CD) diagram matching SAES-UI."""
        cd_res = self._resolve_cd_result(df, **kwargs)
        width_px, height_px = resolve_figure_dimensions(kwargs, default_width=800, default_height=495)
        font_scale = int(kwargs.get("font_scale", 13))
        font_family = str(kwargs.get("font_family", "Inter, -apple-system, sans-serif"))
        palette_name = str(kwargs.get("palette", self.palette))

        ranks = cd_res.ranks
        k = cd_res.k
        cd = cd_res.cd
        cliques = cd_res.cliques
        color_map = get_color_map([r.algorithm for r in ranks], palette_name, kwargs.get("custom_colors"))

        max_algo_chars = max((len(r.algorithm) for r in ranks), default=8)
        label_total_chars = max_algo_chars + 9
        approx_char_px = font_scale * 0.65
        required_side_px = max(165, math.ceil(label_total_chars * approx_char_px) + 28)

        plot_inner_width_px = max(240, width_px - 2 * required_side_px - 48)
        units_per_px = max(1, k - 1) / plot_inner_width_px
        side_pad_units = required_side_px * units_per_px
        leader_gap_units = max(0.18, 22 * units_per_px)
        left_edge_x = 1.0 - leader_gap_units
        right_edge_x = float(k) + leader_gap_units

        fig = go.Figure()
        y_first_leader = -0.15 - len(cliques) * 0.06
        n_left = math.ceil(len(ranks) / 2)
        max_row_idx = max(0, n_left - 1)
        y_step = max(0.14, 0.36 / max_row_idx) if max_row_idx > 0 else 0.15

        for idx, item in enumerate(ranks):
            color = color_map[item.algorithm]
            is_left = idx < n_left
            row_idx = idx if is_left else (len(ranks) - 1 - idx)
            y_leader = y_first_leader - row_idx * y_step

            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    name=f"{item.algorithm} (Rank {item.avg_rank:.2f})",
                    x=[item.avg_rank, item.avg_rank],
                    y=[0, y_leader],
                    marker={
                        "color": color,
                        "size": [10, 7],
                        "symbol": ["circle", "diamond"],
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                    hovertemplate=(
                        f"<b>{item.algorithm}</b><br>"
                        f"Average Rank: <b>{item.avg_rank:.3f}</b><br>"
                        f"Mean {cd_res.metric}: {item.mean_value:.4f}<extra></extra>"
                    ),
                )
            )

        shapes: list[dict[str, Any]] = [
            {
                "type": "line",
                "x0": 1,
                "x1": k,
                "y0": 0,
                "y1": 0,
                "line": {"color": "#0f172a", "width": 2.2},
            }
        ]
        annotations: list[dict[str, Any]] = []

        t = 1.0
        while t <= k + 1e-9:
            is_major = abs(t - round(t)) < 1e-9
            shapes.append(
                {
                    "type": "line",
                    "x0": t,
                    "x1": t,
                    "y0": 0,
                    "y1": 0.055 if is_major else 0.03,
                    "line": {"color": "#0f172a", "width": 1.8 if is_major else 1.0},
                }
            )
            if is_major:
                annotations.append(
                    {
                        "x": t,
                        "y": 0.07,
                        "text": f"<b>{int(round(t))}</b>",
                        "showarrow": False,
                        "yanchor": "bottom",
                        "font": {"family": font_family, "size": font_scale, "color": "#0f172a"},
                    }
                )
            t += 0.5

        annotations.append(
            {
                "x": 1,
                "y": 0.165,
                "text": "<b>← Better (Rank 1)</b>",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {"family": font_family, "size": font_scale - 1, "color": "#334155"},
            }
        )

        cd_span = min(cd, float(k - 1))
        cd_start = max(1.0, float(k) - cd_span)
        cd_end = float(k)
        y_cd_bar = 0.29
        shapes.extend(
            [
                {
                    "type": "line",
                    "x0": cd_start,
                    "x1": cd_end,
                    "y0": y_cd_bar,
                    "y1": y_cd_bar,
                    "line": {"color": "#ef4444", "width": 3},
                },
                {
                    "type": "line",
                    "x0": cd_start,
                    "x1": cd_start,
                    "y0": y_cd_bar - 0.025,
                    "y1": y_cd_bar + 0.025,
                    "line": {"color": "#ef4444", "width": 2},
                },
                {
                    "type": "line",
                    "x0": cd_end,
                    "x1": cd_end,
                    "y0": y_cd_bar - 0.025,
                    "y1": y_cd_bar + 0.025,
                    "line": {"color": "#ef4444", "width": 2},
                },
            ]
        )
        annotations.append(
            {
                "x": (cd_start + cd_end) / 2.0,
                "y": y_cd_bar + 0.032,
                "text": f"<b>CD = {cd:.2f}</b> (α = {cd_res.alpha})",
                "showarrow": False,
                "yanchor": "bottom",
                "font": {"family": font_family, "size": font_scale - 1, "color": "#ef4444"},
            }
        )

        for c_idx, (r_start, r_end) in enumerate(cliques):
            y_clique = -0.08 - c_idx * 0.055
            shapes.append(
                {
                    "type": "line",
                    "x0": r_start - 0.03,
                    "x1": r_end + 0.03,
                    "y0": y_clique,
                    "y1": y_clique,
                    "line": {"color": "#0f172a", "width": 5.5},
                }
            )

        for idx, item in enumerate(ranks):
            color = color_map[item.algorithm]
            is_left = idx < n_left
            row_idx = idx if is_left else (len(ranks) - 1 - idx)
            y_leader = y_first_leader - row_idx * y_step
            x_end = left_edge_x if is_left else right_edge_x

            shapes.extend(
                [
                    {
                        "type": "line",
                        "x0": item.avg_rank,
                        "x1": item.avg_rank,
                        "y0": 0,
                        "y1": y_leader,
                        "line": {"color": color, "width": 1.5, "dash": "dot"},
                    },
                    {
                        "type": "line",
                        "x0": item.avg_rank,
                        "x1": x_end,
                        "y0": y_leader,
                        "y1": y_leader,
                        "line": {"color": color, "width": 1.6},
                    },
                ]
            )
            label_html = (
                f"<b>{item.algorithm}</b> ({item.avg_rank:.2f})"
                if is_left
                else f"({item.avg_rank:.2f}) <b>{item.algorithm}</b>"
            )
            annotations.append(
                {
                    "x": x_end,
                    "y": y_leader,
                    "text": label_html,
                    "xshift": -6 if is_left else 6,
                    "showarrow": False,
                    "xanchor": "right" if is_left else "left",
                    "yanchor": "middle",
                    "font": {"family": font_family, "size": font_scale, "color": "#0f172a"},
                }
            )

        min_bottom_y = (y_first_leader - max_row_idx * y_step) - 0.055

        title_y = 1.0 - (22.0 / max(320.0, float(height_px)))
        fig.update_layout(
            width=width_px,
            height=height_px,
            autosize=False,
            title={
                "text": (
                    f"<b>{cd_res.metrics_label}</b>  •  {cd_res.instances_label}<br>"
                    f"<span style='font-size:{font_scale - 1}px'>{cd_res.params_label}</span>"
                ),
                "x": 0.5,
                "xanchor": "center",
                "y": title_y,
                "yanchor": "top",
                "font": {"family": font_family, "size": font_scale + 1, "color": "#0f172a"},
            },
            font={"family": font_family, "size": font_scale, "color": "#334155"},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis={"range": [1.0 - side_pad_units, float(k) + side_pad_units], "visible": False, "fixedrange": True},
            yaxis={"range": [min_bottom_y, 0.36], "visible": False, "fixedrange": True},
            shapes=shapes,
            annotations=annotations,
            showlegend=bool(kwargs.get("show_legend", False)),
            margin={"l": 24, "r": 24, "t": 68, "b": 24},
        )
        return fig
