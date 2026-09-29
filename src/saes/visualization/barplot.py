"""Bar plot generator with error bounds for algorithm performance comparison."""

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from saes.domain.exceptions import VisualizationError
from saes.visualization.base import Plotter
from saes.visualization.utils import format_plot_title, get_color_map


class BarPlotGenerator(Plotter):
    """Generates bar plots with mean/median metric values and confidence/error bars."""

    def __init__(self, style: str = "whitegrid", palette: str = "Set2") -> None:
        self.style = style
        self.palette = palette

    @property
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        return "Bar Plot (Mean ± Error)"

    @property
    def description(self) -> str:
        """Short description of what the plot displays."""
        return "Compares mean or median algorithm performance with customizable error bars (std dev, std err)."

    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate a Matplotlib bar plot comparing algorithm values.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting options.

        Returns:
            plt.Figure: Matplotlib figure instance.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for bar plot generation.")

        title = format_plot_title(kwargs, "Algorithm Mean Performance (Bar Plot)")
        metric = kwargs.get("metric", "value")
        error_bar = kwargs.get("error_bar", "sd")  # 'sd', 'se', 'ci'

        try:
            sns.set_theme(style=self.style)
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

            error_opt = "sd" if error_bar == "sd" else ("se" if error_bar == "se" else 95)

            sns.barplot(
                data=df,
                x="algorithm",
                y="value",
                hue="algorithm",
                legend=False,
                palette=self.palette,
                ax=ax,
                errorbar=error_opt,
                capsize=0.1,
            )

            ax.set_xlabel("Algorithm", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Mean Metric: {metric}", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")

            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate bar plot: {str(e)}") from e

    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate an interactive Plotly bar plot comparing algorithm values with error bars.

        Args:
            df: DataFrame containing 'algorithm' and 'value' columns.
            **kwargs: Plotting configuration parameters.

        Returns:
            go.Figure: Plotly figure object.
        """
        if "algorithm" not in df.columns or "value" not in df.columns:
            raise VisualizationError("DataFrame must contain 'algorithm' and 'value' columns for bar plot generation.")

        try:
            metric = kwargs.get("metric", "value")
            title = format_plot_title(kwargs, "Algorithm Mean Performance (Bar Plot)")
            height = kwargs.get("height", 500)
            error_type = kwargs.get("error_bar", "sd")  # 'sd', 'se', 'none'
            stat_type = kwargs.get("stat_type", "mean")  # 'mean', 'median'
            palette_name = kwargs.get("palette", "Plotly")
            custom_colors = kwargs.get("custom_colors", None)
            log_y = kwargs.get("log_y", False)

            algs = list(df["algorithm"].unique())
            color_map = get_color_map(algs, palette_name, custom_colors)

            summary_rows = []
            for alg, group in df.groupby("algorithm"):
                vals = group["value"].to_numpy()
                center = float(np.mean(vals)) if stat_type == "mean" else float(np.median(vals))
                n = len(vals)

                if error_type == "sd":
                    err = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                elif error_type == "se":
                    err = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                else:
                    err = 0.0

                summary_rows.append(
                    {
                        "algorithm": str(alg),
                        "center": center,
                        "error": err,
                        "count": n,
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0,
                    }
                )

            summary_df = pd.DataFrame(summary_rows)

            fig = go.Figure()

            for _, row in summary_df.iterrows():
                alg = row["algorithm"]
                color = color_map.get(alg, "#1f77b4")
                error_dict = dict(type="data", array=[row["error"]], visible=row["error"] > 0)

                hover_text = (
                    f"<b>Algorithm:</b> {alg}<br>"
                    f"<b>{stat_type.capitalize()}:</b> {row['center']:.4f}<br>"
                    f"<b>Std Dev:</b> {row['std']:.4f}<br>"
                    f"<b>Min:</b> {row['min']:.4f} | <b>Max:</b> {row['max']:.4f}<br>"
                    f"<b>Runs (N):</b> {int(row['count'])}"
                )

                fig.add_trace(
                    go.Bar(
                        name=alg,
                        x=[alg],
                        y=[row["center"]],
                        error_y=error_dict,
                        marker_color=color,
                        hoverinfo="text",
                        hovertext=hover_text,
                    )
                )

            fig.update_layout(
                height=height,
                template="plotly_white",
                title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}},
                xaxis_title="Algorithm",
                yaxis_title=f"{stat_type.capitalize()} Metric: {metric}",
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=True,
                legend_title_text="Algorithm",
            )

            if log_y:
                fig.update_yaxes(type="log")

            return fig
        except Exception as e:
            raise VisualizationError(f"Failed to generate interactive bar plot: {str(e)}") from e
