"""Violin plot generator for statistical algorithm distribution comparisons."""

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


class ViolinPlotGenerator(Plotter):
    """Generates violin plots depicting probability density distributions across algorithms."""

    def __init__(self, style: str = "whitegrid", palette: str = "Set2") -> None:
        self.style = style
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Violin Plot"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return "Combines boxplot and kernel density estimation to reveal distribution shape and multimodality."

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a Matplotlib violin plot comparing algorithm values.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting configuration options.

        Returns:
            plt.Figure: Matplotlib figure instance.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError(
                "DataFrame must contain 'algorithm' and 'value' columns for violin plot generation."
            )

        title = format_plot_title(kwargs, "Algorithm Metric Distribution (Violin Plot)")
        metric = kwargs.get("metric", "value")

        try:
            sns.set_theme(style=self.style)
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

            sns.violinplot(
                data=df,
                x="algorithm",
                y="value",
                hue="algorithm",
                legend=False,
                palette=self.palette,
                ax=ax,
                inner="quartile",
                cut=0,
            )

            ax.set_xlabel("Algorithm", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Metric: {metric}", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate violin plot: {str(e)}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly violin plot comparing algorithm values.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting configuration parameters.

        Returns:
            go.Figure: Plotly figure object.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError(
                "DataFrame must contain 'algorithm' and 'value' columns for violin plot generation."
            )

        try:
            metric = kwargs.get("metric", "value")
            title = format_plot_title(kwargs, "Algorithm Metric Distribution (Violin Plot)")
            height = kwargs.get("height", 500)
            show_points = kwargs.get("show_points", True)
            palette_name = kwargs.get("palette", "Plotly")
            custom_colors = kwargs.get("custom_colors", None)
            log_y = kwargs.get("log_y", False)

            algs = list(df["algorithm"].unique())
            color_map = get_color_map(algs, palette_name, custom_colors)

            points_mode = "all" if show_points is True else (show_points if isinstance(show_points, str) else False)
            hover_data = ["run_id"] if "run_id" in df.columns else None

            fig = px.violin(
                df,
                x="algorithm",
                y="value",
                color="algorithm",
                color_discrete_map=color_map,
                points=points_mode,
                box=True,
                hover_data=hover_data,
                title=title,
                labels={"algorithm": "Algorithm", "value": f"Metric: {metric}"},
                log_y=log_y,
            )

            fig.update_traces(pointpos=0, jitter=0.35)

            fig.update_layout(
                height=height,
                template="plotly_white",
                title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
                xaxis_title="Algorithm",
                yaxis_title=f"Metric: {metric}",
                margin=dict(l=40, r=40, t=60, b=40),
                legend_title_text="Algorithm",
            )
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate interactive violin plot: {str(e)}") from e
