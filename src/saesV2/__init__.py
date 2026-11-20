"""SAES: Stochastic Algorithm Evaluation Suite."""

from typing import Any

from saesV2._version import __version__ as __version__
from saesV2.api import (
    analyze,
    analyze_statistical_power,
    compute_performance_profiles,
    explore,
    export_exploration,
    export_plot,
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
from saesV2.domain.exceptions import (
    ConfigurationError,
    InfrastructureError,
    InputError,
    SAESError,
    StatisticalAnalysisError,
    ValidationError,
    VisualizationError,
)
from saesV2.domain.models import (
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


def __getattr__(name: str) -> Any:
    if name == "PlotterRegistry":
        from saesV2.visualization.registry import PlotterRegistry

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
    "export_plot",
    "statistics",
    "profiles",
    "power",
    "reproducibility",
    "compute_performance_profiles",
    "analyze_statistical_power",
    "generate_reproducibility_artifacts",
]

