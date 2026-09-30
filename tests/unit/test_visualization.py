"""Unit tests for visualization module and Plotter implementations."""

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from saes.domain.exceptions import VisualizationError
from saes.visualization.barplot import BarPlotGenerator
from saes.visualization.boxplot import BoxplotGenerator
from saes.visualization.ecdfplot import ECDFPlotGenerator
from saes.visualization.lineplot import LinePlotGenerator
from saes.visualization.registry import PlotterRegistry
from saes.visualization.scatterplot import ScatterPlotGenerator
from saes.visualization.utils import get_color_map, to_hex_color
from saes.visualization.violinplot import ViolinPlotGenerator


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "algorithm": ["NSGAII", "NSGAII", "MOEAD", "MOEAD"],
            "instance": ["ZDT1", "ZDT1", "ZDT1", "ZDT1"],
            "metric": ["HV", "HV", "HV", "HV"],
            "run_id": [0, 1, 0, 1],
            "value": [0.65, 0.67, 0.68, 0.69],
        }
    )


def test_boxplot_generator_success(sample_df: pd.DataFrame) -> None:
    generator = BoxplotGenerator()
    assert generator.name == "Box Plot"
    assert "quartile" in generator.description.lower()

    fig_mpl = generator.plot(sample_df, title="Test Boxplot", metric="HV")
    assert isinstance(fig_mpl, Figure)

    fig_plotly = generator.plot_interactive(sample_df, title="Interactive Boxplot", height=600)
    assert isinstance(fig_plotly, go.Figure)


def test_violinplot_generator_success(sample_df: pd.DataFrame) -> None:
    generator = ViolinPlotGenerator()
    assert generator.name == "Violin Plot"

    fig_mpl = generator.plot(sample_df, metric="HV")
    assert isinstance(fig_mpl, Figure)

    fig_plotly = generator.plot_interactive(sample_df, metric="HV", custom_colors={"NSGAII": "#ff0000"})
    assert isinstance(fig_plotly, go.Figure)


def test_barplot_generator_success(sample_df: pd.DataFrame) -> None:
    generator = BarPlotGenerator()
    assert generator.name == "Bar Plot (Mean ± Error)"

    fig_mpl = generator.plot(sample_df, metric="HV", error_bar="sd")
    assert isinstance(fig_mpl, Figure)

    fig_plotly = generator.plot_interactive(sample_df, metric="HV", stat_type="median", error_bar="se")
    assert isinstance(fig_plotly, go.Figure)


def test_lineplot_generator_success(sample_df: pd.DataFrame) -> None:
    generator = LinePlotGenerator()
    assert generator.name == "Line Plot (Run Trails)"

    fig_mpl = generator.plot(sample_df, metric="HV")
    assert isinstance(fig_mpl, Figure)

    fig_plotly = generator.plot_interactive(sample_df, metric="HV")
    assert isinstance(fig_plotly, go.Figure)


def test_ecdfplot_generator_success(sample_df: pd.DataFrame) -> None:
    generator = ECDFPlotGenerator()
    assert generator.name == "ECDF Plot"

    fig_mpl = generator.plot(sample_df, metric="HV")
    assert isinstance(fig_mpl, Figure)

    fig_plotly = generator.plot_interactive(sample_df, metric="HV")
    assert isinstance(fig_plotly, go.Figure)


def test_scatterplot_generator_success(sample_df: pd.DataFrame) -> None:
    generator = ScatterPlotGenerator()
    assert generator.name == "Scatter / Jitter Plot"

    fig_mpl = generator.plot(sample_df, metric="HV")
    assert isinstance(fig_mpl, Figure)

    fig_plotly = generator.plot_interactive(sample_df, metric="HV")
    assert isinstance(fig_plotly, go.Figure)


def test_missing_columns_raise_visualization_error() -> None:
    invalid_df = pd.DataFrame({"invalid_col": [1, 2, 3]})

    for key in PlotterRegistry.list_available():
        generator = PlotterRegistry.get(key)
        with pytest.raises(VisualizationError, match="algorithm"):
            generator.plot(invalid_df)
        with pytest.raises(VisualizationError, match="algorithm"):
            generator.plot_interactive(invalid_df)


def test_plotter_registry() -> None:
    available = PlotterRegistry.list_available()
    expected_plotters = {"boxplot", "violin", "barplot", "lineplot", "ecdf", "scatter"}
    for p in expected_plotters:
        assert p in available
    assert len(available) == 6

    boxplot_instance = PlotterRegistry.get("boxplot")
    assert isinstance(boxplot_instance, BoxplotGenerator)

    with pytest.raises(VisualizationError, match="Unknown plotter key"):
        PlotterRegistry.get("nonexistent_plotter")


def test_to_hex_color_conversion() -> None:
    assert to_hex_color("rgb(228, 26, 28)") == "#E41A1C"
    assert to_hex_color("rgba(0, 255, 0, 0.5)") == "#00FF00"
    assert to_hex_color("#123456") == "#123456"
    assert to_hex_color("#abc") == "#AABBCC"


def test_custom_color_map() -> None:
    algs = ["NSGAII", "MOEAD"]
    custom = {"NSGAII": "rgb(228, 26, 28)"}
    cmap = get_color_map(algs, palette_name="Set1", custom_colors=custom)
    assert cmap["NSGAII"] == "#E41A1C"
    assert cmap["MOEAD"].startswith("#")
