"""SAES: Stochastic Algorithm Evaluation Suite."""

from typing import Any

from saes.api import (
    analyze,
    analyze_statistical_power,
    compute_performance_profiles,
    explore,
    export_exploration,
    export_table,
    generate_reproducibility_artifacts,
    generate_table,
    load,
    plot,
    power,
    profiles,
    reproducibility,
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
    AlgorithmProfileCurve,
    AlgorithmRankItem,
    BayesianPosteriorItem,
    BayesianRankResult,
    CriticalDistanceResult,
    DataExplorationResult,
    Dataset,
    EnvironmentManifest,
    GlobalMetaRankingItem,
    PerformanceProfilesResult,
    PostHocPowerItem,
    PowerAnalysisResult,
    PowerCurveSeries,
    ReproducibilityResult,
    SampleSizeRecommendation,
    StatisticalResult,
    StatisticalTableResult,
    ValidationReport,
    WinTieLossCell,
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
    "StatisticalTableResult",
    "DataExplorationResult",
    "ValidationReport",
    "AlgorithmRankItem",
    "CriticalDistanceResult",
    "BayesianPosteriorItem",
    "BayesianRankResult",
    "PerformanceProfilesResult",
    "AlgorithmProfileCurve",
    "WinTieLossCell",
    "GlobalMetaRankingItem",
    "PowerAnalysisResult",
    "SampleSizeRecommendation",
    "PowerCurveSeries",
    "PostHocPowerItem",
    "ReproducibilityResult",
    "EnvironmentManifest",
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
    "generate_table",
    "export_table",
    "explore",
    "export_exploration",
    "plot",
    "statistics",
    "profiles",
    "power",
    "reproducibility",
    "compute_performance_profiles",
    "analyze_statistical_power",
    "generate_reproducibility_artifacts",
]
