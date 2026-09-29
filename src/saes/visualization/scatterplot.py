"""Scatter / Strip plot generator for individual run observations."""

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from saes.domain.exceptions import VisualizationError
from saes.visualization.base import Plotter
from saes.visualization.utils import format_plot_title, get_color_map


class ScatterPlotGenerator(Plotter):
    """Generates jittered scatter plots displaying individual run values per algorithm."""

    def __init__(self, style: str = "whitegrid", palette: str = "Set2") -> None:
        self.style = style
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Scatter / Jitter Plot"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return "Displays individual run observations as jittered scatter points for raw data inspection."

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a Matplotlib stripplot.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting options.

        Returns:
            plt.Figure: Matplotlib figure instance.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError(
                "DataFrame must contain 'algorithm' and 'value' columns for scatter plot generation."
            )

        title = format_plot_title(kwargs, "Algorithm Run Observations (Scatter / Jitter)")
        metric = kwargs.get("metric", "value")
        point_size = kwargs.get("point_size", 6)
        log_y = kwargs.get("log_y", False)
        show_legend = kwargs.get("show_legend", True)

        palette_opt = kwargs.get("custom_colors") or kwargs.get("palette", self.palette)
        if isinstance(palette_opt, str):
            algs = list(df["algorithm"].unique())
            palette_opt = get_color_map(algs, palette_opt)

        try:
            sns.set_theme(style=self.style)
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

            sns.stripplot(
                data=df,
                x="algorithm",
                y="value",
                hue="algorithm",
                legend=show_legend,
                palette=palette_opt,
                jitter=0.25,
                size=point_size,
                alpha=0.8,
                ax=ax,
            )

            if log_y:
                ax.set_yscale("log")

            ax.set_xlabel("Algorithm", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Metric: {metric}", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate scatter plot: {str(e)}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly strip/scatter plot.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting configuration options.

        Returns:
            go.Figure: Plotly figure object.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError(
                "DataFrame must contain 'algorithm' and 'value' columns for scatter plot generation."
            )

        try:
            metric = kwargs.get("metric", "value")
            title = format_plot_title(kwargs, "Algorithm Run Observations (Scatter / Jitter)")
            height = kwargs.get("height", 500)
            palette_name = kwargs.get("palette", "Plotly")
            custom_colors = kwargs.get("custom_colors", None)
            log_y = kwargs.get("log_y", False)

            algs = list(df["algorithm"].unique())
            color_map = get_color_map(algs, palette_name, custom_colors)

            hover_data = ["run_id"] if "run_id" in df.columns else None

            fig = px.strip(
                df,
                x="algorithm",
                y="value",
                color="algorithm",
                color_discrete_map=color_map,
                hover_data=hover_data,
                title=title,
                labels={"algorithm": "Algorithm", "value": f"Metric: {metric}"},
                log_y=log_y,
            )

            point_size = kwargs.get("point_size", 8)
            fig.update_traces(marker=dict(size=point_size, opacity=0.8))

            show_legend = kwargs.get("show_legend", True)

            fig.update_layout(
                height=height,
                template="plotly_white",
                title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
                xaxis_title="Algorithm",
                yaxis_title=f"Metric: {metric}",
                margin=dict(l=40, r=40, t=60, b=40),
                legend_title_text="Algorithm",
                showlegend=show_legend,
            )
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate interactive scatter plot: {str(e)}") from e
