"""Visualization package for plotting statistical distribution charts."""

from saes.visualization.barplot import BarPlotGenerator
from saes.visualization.base import Plotter
from saes.visualization.bayesianplot import BayesianRankPlotGenerator
from saes.visualization.boxplot import BoxplotGenerator
from saes.visualization.cdplot import CriticalDistancePlotGenerator
from saes.visualization.ecdfplot import ECDFPlotGenerator
from saes.visualization.lineplot import LinePlotGenerator
from saes.visualization.registry import PlotterRegistry
from saes.visualization.scatterplot import ScatterPlotGenerator
from saes.visualization.violinplot import ViolinPlotGenerator

__all__ = [
    "Plotter",
    "BoxplotGenerator",
    "ViolinPlotGenerator",
    "CriticalDistancePlotGenerator",
    "BayesianRankPlotGenerator",
    "BarPlotGenerator",
    "LinePlotGenerator",
    "ECDFPlotGenerator",
    "ScatterPlotGenerator",
    "PlotterRegistry",
]
