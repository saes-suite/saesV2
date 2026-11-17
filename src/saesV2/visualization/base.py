"""Abstract base class for data visualization plotters."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure


class Plotter(ABC):
    """Abstract Strategy interface for generating statistical distribution plots."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name of the plot type."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the plot displays."""
        pass

    @abstractmethod
    def plot(self, df: pd.DataFrame, **kwargs: Any) -> Figure:
        """Generate Matplotlib static figure from input DataFrame.

        Args:
            df: Filtered dataset DataFrame.
            **kwargs: Plotting configuration parameters (title, metric, instance, etc.).

        Returns:
            matplotlib.figure.Figure: Generated plot figure.
        """
        pass

    @abstractmethod
    def plot_interactive(self, df: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Generate Plotly interactive figure from input DataFrame.

        Args:
            df: Filtered dataset DataFrame.
            **kwargs: Plotting parameters (title, metric, instance, height, palette, etc.).

        Returns:
            plotly.graph_objects.Figure: Generated interactive plot figure.
        """
        pass
