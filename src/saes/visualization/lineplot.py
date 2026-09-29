"""Line plot generator for tracking metric runs across run IDs."""

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


class LinePlotGenerator(Plotter):
    """Generates line plots showing algorithm performance trails across individual runs."""

    def __init__(self, style: str = "whitegrid", palette: str = "Set2") -> None:
        self.style = style
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Line Plot (Run Trails)"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return "Plots metric trajectory across run IDs to evaluate stability and variability across seeds."

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a Matplotlib line plot comparing run series.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting options.

        Returns:
            plt.Figure: Matplotlib figure instance.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for line plot generation.")

        title = format_plot_title(kwargs, "Algorithm Metric Trails across Runs")
        metric = kwargs.get("metric", "value")

        plot_df = df.copy()
        if "run_id" not in plot_df.columns:
            plot_df["run_id"] = plot_df.groupby("algorithm").cumcount()

        try:
            sns.set_theme(style=self.style)
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

            sns.lineplot(
                data=plot_df,
                x="run_id",
                y="value",
                hue="algorithm",
                marker="o",
                palette=self.palette,
                ax=ax,
            )

            ax.set_xlabel("Run ID", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Metric: {metric}", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate line plot: {str(e)}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly line plot comparing run series.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting configuration options.

        Returns:
            go.Figure: Plotly figure object.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for line plot generation.")

        try:
            metric = kwargs.get("metric", "value")
            title = format_plot_title(kwargs, "Algorithm Metric Trails across Runs")
            height = kwargs.get("height", 500)
            palette_name = kwargs.get("palette", "Plotly")
            custom_colors = kwargs.get("custom_colors", None)
            log_y = kwargs.get("log_y", False)

            plot_df = df.copy()
            if "run_id" not in plot_df.columns:
                plot_df["run_id"] = plot_df.groupby("algorithm").cumcount()

            plot_df = plot_df.sort_values(by=["algorithm", "run_id"])

            algs = list(plot_df["algorithm"].unique())
            color_map = get_color_map(algs, palette_name, custom_colors)

            fig = px.line(
                plot_df,
                x="run_id",
                y="value",
                color="algorithm",
                markers=True,
                color_discrete_map=color_map,
                hover_data=["run_id", "value"],
                title=title,
                labels={"run_id": "Run ID", "value": f"Metric: {metric}", "algorithm": "Algorithm"},
                log_y=log_y,
            )

            fig.update_layout(
                height=height,
                template="plotly_white",
                title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
                xaxis_title="Run ID",
                yaxis_title=f"Metric: {metric}",
                margin=dict(l=40, r=40, t=60, b=40),
                legend_title_text="Algorithm",
            )
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate interactive line plot: {str(e)}") from e
