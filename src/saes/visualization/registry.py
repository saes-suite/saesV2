"""Plotter registry for dynamic plotter discovery and extensible visualization."""

from typing import Any

from saes.domain.exceptions import VisualizationError
from saes.visualization.base import Plotter
from saes.visualization.boxplot import BoxplotGenerator
from saes.visualization.scatterplot import ScatterPlotGenerator


class PlotterRegistry:
    """Central registry managing statistical plot generators."""

    _registry: dict[str, type[Plotter]] = {
        "boxplot": BoxplotGenerator,
        "scatter": ScatterPlotGenerator,
    }

    @classmethod
    def register(cls, key: str, plotter_cls: type[Plotter]) -> None:
        """Register a new Plotter class.

        Args:
            key: Unique string identifier for the plot type.
            plotter_cls: Plotter subclass.
        """
        cls._registry[key.lower()] = plotter_cls

    @classmethod
    def get(cls, key: str, **kwargs: Any) -> Plotter:
        """Instantiate a Plotter by key.

        Args:
            key: Plotter key identifier.
            **kwargs: Arguments to pass to plotter constructor.

        Returns:
            Plotter instance.
        """
        key_lower = key.lower()
        if key_lower not in cls._registry:
            valid_keys = ", ".join(cls._registry.keys())
            raise VisualizationError(f"Unknown plotter key '{key}'. Available options: {valid_keys}")
        return cls._registry[key_lower](**kwargs)

    @classmethod
    def list_available(cls) -> dict[str, dict[str, str]]:
        """List all available registered plotters with metadata.

        Returns:
            dict mapping key -> {'name': str, 'description': str}
        """
        result: dict[str, dict[str, str]] = {}
        for key, cls_type in cls._registry.items():
            instance = cls_type()
            result[key] = {
                "name": instance.name,
                "description": instance.description,
            }
        return result
