"""SAES: Stochastic Algorithm Evaluation Suite."""

from typing import Any

from saes.api import (
    analyze,
    load,
    plot,
    statistics,
    validate,
)
from saes.domain.exceptions import (
    ConfigurationError,
    InfrastructureError,
    InputError,
    SAESError,
    StatisticalAnalysisError,
    ValidationError,
    VisualizationError,
)
from saes.domain.models import (
    AlgorithmRankItem,
    BayesianPosteriorItem,
    BayesianRankResult,
    CriticalDistanceResult,
    Dataset,
    StatisticalResult,
    ValidationReport,
)

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "PlotterRegistry":
        from saes.visualization.registry import PlotterRegistry

        return PlotterRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Dataset",
    "StatisticalResult",
    "ValidationReport",
    "AlgorithmRankItem",
    "CriticalDistanceResult",
    "BayesianPosteriorItem",
    "BayesianRankResult",
    "PlotterRegistry",
    "SAESError",
    "InputError",
    "ValidationError",
    "ConfigurationError",
    "StatisticalAnalysisError",
    "VisualizationError",
    "InfrastructureError",
    "load",
    "validate",
    "analyze",
    "plot",
    "statistics",
]
