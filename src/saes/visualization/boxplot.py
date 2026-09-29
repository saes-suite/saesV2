"""Boxplot generator for statistical algorithm metric comparisons."""

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


class BoxplotGenerator(Plotter):
    """Generates publication-quality boxplots comparing algorithm metric performance."""

    def __init__(self, style: str = "whitegrid", palette: str = "Set2") -> None:
        self.style = style
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Box Plot"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return "Shows quartile distributions, median, spread, and outliers across algorithms."

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a Matplotlib boxplot comparing algorithm values.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs:
                title (str): Custom plot title.
                instance (str): Instance name for title.
                metric (str): Metric name for axis label.
                show_points (bool): Overlay individual run observations (default True).

        Returns:
            plt.Figure: Matplotlib figure instance.

        Raises:
            VisualizationError: If required columns are missing or plotting fails.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for boxplot generation.")

        title = kwargs.get("title")
        instance = kwargs.get("instance", "")
        metric = kwargs.get("metric", "value")
        show_points = kwargs.get("show_points", True)
        box_width = kwargs.get("box_width", 0.5)
        point_size = kwargs.get("point_size", 5)
        log_y = kwargs.get("log_y", False)
        show_legend = kwargs.get("show_legend", True)

        palette_opt = kwargs.get("custom_colors") or kwargs.get("palette", self.palette)
        if isinstance(palette_opt, str):
            algs = list(df["algorithm"].unique())
            palette_opt = get_color_map(algs, palette_opt)

        try:
            sns.set_theme(style=self.style)
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

            sns.boxplot(
                data=df,
                x="algorithm",
                y="value",
                hue="algorithm",
                legend=show_legend,
                palette=palette_opt,
                ax=ax,
                width=box_width,
                boxprops={"alpha": 0.8},
            )

            if show_points:
                sns.stripplot(
                    data=df,
                    x="algorithm",
                    y="value",
                    color="black",
                    alpha=0.6,
                    jitter=0.2,
                    size=point_size,
                    ax=ax,
                )

            if log_y:
                ax.set_yscale("log")

            ax.set_xlabel("Algorithm", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Metric: {metric}", fontsize=11, fontweight="bold")

            if title:
                ax.set_title(title, fontsize=12, fontweight="bold")
            elif instance:
                ax.set_title(f"Algorithm Metric Distribution ({instance} - {metric})", fontsize=12, fontweight="bold")
            else:
                ax.set_title(f"Algorithm Metric Distribution ({metric})", fontsize=12, fontweight="bold")

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate boxplot: {str(e)}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly boxplot comparing algorithm values.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs:
                title (str): Custom plot title.
                instance (str): Instance name.
                metric (str): Metric name for axis label.
                show_points (bool): Show individual data points ('all', 'outliers', or False).
                height (int): Plot height in pixels.
                palette (str): Color palette name.
                custom_colors (dict): Per-algorithm color overrides.
                log_y (bool): Logarithmic y-axis.

        Returns:
            go.Figure: Plotly figure object.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for boxplot generation.")

        try:
            metric = kwargs.get("metric", "value")
            title = format_plot_title(kwargs, "Algorithm Metric Distribution (Box Plot)")
            height = kwargs.get("height", 500)
            show_points = kwargs.get("show_points", True)
            palette_name = kwargs.get("palette", "Plotly")
            custom_colors = kwargs.get("custom_colors", None)
            log_y = kwargs.get("log_y", False)

            algs = list(df["algorithm"].unique())
            color_map = get_color_map(algs, palette_name, custom_colors)

            points_mode = "all" if show_points is True else (show_points if isinstance(show_points, str) else False)

            hover_data = ["run_id"] if "run_id" in df.columns else None

            fig = px.box(
                df,
                x="algorithm",
                y="value",
                color="algorithm",
                color_discrete_map=color_map,
                points=points_mode,
                hover_data=hover_data,
                title=title,
                labels={"algorithm": "Algorithm", "value": f"Metric: {metric}"},
                log_y=log_y,
            )

            box_width = kwargs.get("box_width", None)
            point_size = kwargs.get("point_size", None)

            trace_kwargs: dict[str, Any] = {"pointpos": 0, "jitter": 0.35}
            if box_width is not None:
                trace_kwargs["width"] = box_width
            if point_size is not None:
                trace_kwargs["marker_size"] = point_size

            fig.update_traces(**trace_kwargs)

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
            raise VisualizationError(f"Failed to generate interactive boxplot: {str(e)}") from e
