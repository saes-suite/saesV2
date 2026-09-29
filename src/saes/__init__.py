"""SAES: Stochastic Algorithm Evaluation Suite."""

from saes.api import analyze, load, statistics, validate
from saes.domain.exceptions import (
    ConfigurationError,
    InfrastructureError,
    InputError,
    SAESError,
    StatisticalAnalysisError,
    ValidationError,
    VisualizationError,
)
from saes.domain.models import Dataset, StatisticalResult, ValidationReport

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "StatisticalResult",
    "ValidationReport",
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
    "statistics",
]
