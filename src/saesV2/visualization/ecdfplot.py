"""ECDF (Empirical Cumulative Distribution Function) plot generator."""

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from saesV2.domain.exceptions import VisualizationError
from saesV2.visualization.base import Plotter
from saesV2.visualization.utils import format_plot_title, get_color_map


class ECDFPlotGenerator(Plotter):
    """Generates empirical cumulative distribution function (ECDF) curves per algorithm."""

    def __init__(self, style: str = "whitegrid", palette: str = "Set2") -> None:
        self.style = style
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "ECDF Plot"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return "Empirical cumulative distribution function showing proportion of runs reaching performance levels."

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a Matplotlib ECDF plot.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting options.

        Returns:
            plt.Figure: Matplotlib figure instance.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for ECDF plot generation.")

        title = format_plot_title(kwargs, "Empirical Cumulative Distribution (ECDF)")
        metric = kwargs.get("metric", "value")

        try:
            sns.set_theme(style=self.style)
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

            sns.ecdfplot(
                data=df,
                x="value",
                hue="algorithm",
                palette=self.palette,
                ax=ax,
            )

            ax.set_xlabel(f"Metric: {metric}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Cumulative Proportion P(X ≤ x)", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate ECDF plot: {str(e)}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly ECDF plot.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting configuration options.

        Returns:
            go.Figure: Plotly figure object.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for ECDF plot generation.")

        try:
            metric = kwargs.get("metric", "value")
            title = format_plot_title(kwargs, "Empirical Cumulative Distribution (ECDF)")
            height = kwargs.get("height", 500)
            palette_name = kwargs.get("palette", "Plotly")
            custom_colors = kwargs.get("custom_colors", None)
            log_y = kwargs.get("log_y", False)

            algs = list(df["algorithm"].unique())
            color_map = get_color_map(algs, palette_name, custom_colors)

            fig = px.ecdf(
                df,
                x="value",
                color="algorithm",
                color_discrete_map=color_map,
                title=title,
                labels={"value": f"Metric: {metric}", "algorithm": "Algorithm"},
                log_y=log_y,
            )

            fig.update_layout(
                height=height,
                template="plotly_white",
                title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
                xaxis_title=f"Metric: {metric}",
                yaxis_title="Cumulative Proportion P(X ≤ x)",
                margin=dict(l=40, r=40, t=60, b=40),
                legend_title_text="Algorithm",
            )
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate interactive ECDF plot: {str(e)}") from e
