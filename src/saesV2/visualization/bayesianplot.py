"""Bayesian Signed-Rank / Sign-Test Simplex ROPE plot generator (Benavoli et al. 2017)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from saesV2.domain.exceptions import VisualizationError
from saesV2.domain.models import BayesianRankResult
from saesV2.stats.ranking import BayesianRankingTest
from saesV2.visualization.base import Plotter
from saesV2.visualization.utils import resolve_figure_dimensions

# Density colorscale matching baycomp / Benavoli et al. (2017) Simplex plot:
# Low density (periphery) = dark navy blue -> High density (core) = bright ice blue / white-cyan
BAYESIAN_SIMPLEX_COLORSCALE: list[list[Any]] = [
    [0.0, "#08306b"],
    [0.25, "#08519c"],
    [0.50, "#2171b5"],
    [0.72, "#4292c6"],
    [0.88, "#9ecae1"],
    [1.0, "#e0f3f8"],
]


class BayesianRankPlotGenerator(Plotter):
    """Generates Bayesian Non-Parametric Simplex (Ternary ROPE) posterior probability plots."""

    def __init__(self, palette: str = "academic") -> None:
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Bayesian Posterior Rank Plot"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return (
            "Displays the Benavoli et al. (2017) Bayesian Simplex (Ternary ROPE) posterior cloud "
            "comparing two algorithms across p(left), p(rope), and p(right) decision regions."
        )

    def _resolve_bayesian_result(self, df: pd.DataFrame, **kwargs: Any) -> BayesianRankResult:
        """Obtain BayesianRankResult from kwargs or compute it from DataFrame."""
        existing = kwargs.get("bayesian_result")
        if isinstance(existing, BayesianRankResult):
            return existing

        required_cols = {"algorithm", "value"}
        if not required_cols.issubset(df.columns):
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for Bayesian Rank plot.")

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

        scope = str(kwargs.get("scope", "all-instances"))
        instance = kwargs.get("instance")
        rope_percent = float(kwargs.get("rope_percent", 1.0))
        algo_a = kwargs.get("algo_a")
        algo_b = kwargs.get("algo_b")
        method = str(kwargs.get("method", "signed-rank"))

        try:
            return BayesianRankingTest().run(
                df=work_df,
                metric=metric,
                scope=scope,
                instance=str(instance) if instance else None,
                rope_percent=rope_percent,
                algo_a=str(algo_a) if algo_a else None,
                algo_b=str(algo_b) if algo_b else None,
                method=method,
            )
        except Exception as e:
            raise VisualizationError(f"Failed to compute Bayesian Simplex posterior: {e!s}") from e

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a publication-grade Matplotlib Bayesian Simplex (Ternary ROPE) plot."""
        bayes_res = self._resolve_bayesian_result(df, **kwargs)
        width_px, height_px = resolve_figure_dimensions(kwargs, default_width=680, default_height=510)
        figsize = kwargs.get("figsize", (width_px / 100.0, height_px / 100.0))

        try:
            fig, ax = plt.subplots(figsize=figsize, dpi=150)

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "bayesian_simplex_blues",
                [c[1] for c in BAYESIAN_SIMPLEX_COLORSCALE],
            )

            # 1. Monte Carlo posterior sample cloud colored by local KDE density
            ax.scatter(
                bayes_res.points_x,
                bayes_res.points_y,
                c=bayes_res.points_density,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                s=14,
                alpha=0.90,
                edgecolors="none",
                zorder=2,
            )

            # 2. Internal orange ROPE decision boundary segments meeting at the centroid
            for x0, y0, x1, y1 in bayes_res.boundary_lines:
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color="#FFA500",
                    linewidth=3.0,
                    solid_capstyle="round",
                    zorder=4,
                )

            # 3. Outer equilateral triangle border in orange
            ax.plot(
                bayes_res.triangle_x,
                bayes_res.triangle_y,
                color="#FFA500",
                linewidth=3.4,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )

            # 4. Vertex probability labels matching Benavoli et al. (2017) / baycomp
            top_y = bayes_res.triangle_y[2] if len(bayes_res.triangle_y) > 2 else 0.866025
            ax.text(
                0.5,
                top_y + 0.032,
                bayes_res.label_rope,
                ha="center",
                va="bottom",
                fontsize=12,
                color="#262626",
                fontweight="medium",
            )
            ax.text(
                0.0,
                -0.042,
                bayes_res.label_left,
                ha="center",
                va="top",
                fontsize=12,
                color="#262626",
                fontweight="medium",
            )
            ax.text(
                1.0,
                -0.042,
                bayes_res.label_right,
                ha="center",
                va="top",
                fontsize=12,
                color="#262626",
                fontweight="medium",
            )

            ax.set_aspect("equal")
            ax.set_xlim(-0.24, 1.24)
            ax.set_ylim(-0.12, top_y + 0.14)
            ax.axis("off")

            default_title = f"{bayes_res.metrics_label}  •  {bayes_res.instances_label}\n{bayes_res.params_label}"
            title = kwargs.get("title", default_title)
            ax.set_title(str(title), fontsize=10, fontweight="bold", pad=18)

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate Matplotlib Bayesian Simplex plot: {e!s}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly Bayesian Simplex (Ternary ROPE) plot matching SAES-UI."""
        bayes_res = self._resolve_bayesian_result(df, **kwargs)
        width_px, height_px = resolve_figure_dimensions(kwargs, default_width=800, default_height=495)
        font_scale = int(kwargs.get("font_scale", 13))
        font_family = str(kwargs.get("font_family", "Inter, -apple-system, sans-serif"))

        fig = go.Figure()

        customdata = list(
            zip(
                bayes_res.points_p_left,
                bayes_res.points_p_rope,
                bayes_res.points_p_right,
                strict=False,
            )
        )

        # 1. Posterior Monte Carlo cloud colored by KDE density
        fig.add_trace(
            go.Scatter(
                x=bayes_res.points_x,
                y=bayes_res.points_y,
                mode="markers",
                name="Posterior Samples",
                customdata=customdata,
                marker={
                    "size": 5.5,
                    "color": bayes_res.points_density,
                    "colorscale": BAYESIAN_SIMPLEX_COLORSCALE,
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "opacity": 0.90,
                    "showscale": False,
                },
                hovertemplate=(
                    f"<b>Posterior Dirichlet Sample</b><br>"
                    f"P({bayes_res.algo_a}): <b>%{{customdata[0]:.3f}}</b><br>"
                    f"P(rope): <b>%{{customdata[1]:.3f}}</b><br>"
                    f"P({bayes_res.algo_b}): <b>%{{customdata[2]:.3f}}</b><extra></extra>"
                ),
                showlegend=False,
            )
        )

        # 2. Internal orange ROPE decision region boundaries
        for x0, y0, x1, y1 in bayes_res.boundary_lines:
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"color": "#FFA500", "width": 3.6},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 3. Outer equilateral triangle border in orange
        fig.add_trace(
            go.Scatter(
                x=bayes_res.triangle_x,
                y=bayes_res.triangle_y,
                mode="lines",
                line={"color": "#FFA500", "width": 4.0},
                hoverinfo="skip",
                showlegend=False,
            )
        )

        top_y = bayes_res.triangle_y[2] if len(bayes_res.triangle_y) > 2 else 0.866025
        annotations: list[dict[str, Any]] = [
            {
                "x": 0.5,
                "y": top_y + 0.035,
                "text": bayes_res.label_rope,
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "bottom",
                "font": {"family": font_family, "size": font_scale + 2, "color": "#262626"},
            },
            {
                "x": 0.0,
                "y": -0.038,
                "text": bayes_res.label_left,
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"family": font_family, "size": font_scale + 2, "color": "#262626"},
            },
            {
                "x": 1.0,
                "y": -0.038,
                "text": bayes_res.label_right,
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"family": font_family, "size": font_scale + 2, "color": "#262626"},
            },
        ]

        top_margin = round(48 + font_scale * 2.3)
        bottom_margin = 24
        plot_w = max(200.0, float(width_px - 48))
        plot_h = max(200.0, float(height_px - top_margin - bottom_margin))
        y_max = top_y + 0.095
        span_y = y_max - (-0.095)
        span_x = span_y * (plot_w / plot_h)
        min_span_x = max(1.32, 1.14 + font_scale * 0.014)
        if span_x < min_span_x:
            span_x = min_span_x
            span_y = span_x * (plot_h / plot_w)
        title_y = 1.0 - ((top_margin * 0.48) / max(320.0, float(height_px)))

        fig.update_layout(
            width=width_px,
            height=height_px,
            autosize=False,
            title={
                "text": (
                    f"<b>{bayes_res.metrics_label}</b>  •  {bayes_res.instances_label}<br>"
                    f"<span style='font-size:{font_scale - 1}px'>{bayes_res.params_label}</span>"
                ),
                "x": 0.5,
                "xanchor": "center",
                "y": title_y,
                "yanchor": "middle",
                "font": {"family": font_family, "size": font_scale, "color": "#262626"},
            },
            font={"family": font_family, "size": font_scale, "color": "#262626"},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis={
                "range": [0.5 - span_x / 2.0, 0.5 + span_x / 2.0],
                "visible": False,
                "fixedrange": True,
            },
            yaxis={
                "range": [y_max - span_y, y_max],
                "visible": False,
                "fixedrange": True,
            },
            annotations=annotations,
            showlegend=False,
            margin={"l": 24, "r": 24, "t": top_margin, "b": bottom_margin},
        )
        return fig
