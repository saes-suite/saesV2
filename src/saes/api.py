"""Public API surface for SAES library."""

from typing import Any

import numpy as np

from saes.app.pipeline import AnalysisPipeline, AnalysisReport
from saes.domain.models import AssumptionResult, Dataset, StatisticalResult, ValidationReport
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def load(source: Any) -> Dataset:
    """Load a CSV dataset into a SAES Dataset object.

    Args:
        source: Filepath or file-like buffer.

    Returns:
        Dataset entity.
    """
    pipeline = AnalysisPipeline()
    return pipeline.load_csv(source)


def validate(dataset: Dataset) -> ValidationReport:
    """Validate dataset structure, schema, and types.

    Args:
        dataset: Dataset instance.

    Returns:
        ValidationReport object.
    """
    pipeline = AnalysisPipeline()
    return pipeline.validate(dataset)


def analyze(
    dataset: Dataset,
    alpha: float = 0.05,
    generate_plots: bool = True,
) -> AnalysisReport:
    """Run comprehensive automated statistical analysis across dataset.

    Args:
        dataset: Validated Dataset instance.
        alpha: Significance level (default 0.05).
        generate_plots: Whether to generate distribution boxplots (default True).

    Returns:
        AnalysisReport: Comprehensive report of all statistical tests and plots.
    """
    pipeline = AnalysisPipeline(alpha=alpha)
    return pipeline.analyze(dataset, generate_plots=generate_plots)


# Low-level programmatic statistics sub-namespace
class StatisticsNamespace:
    """Programmatic statistical test API namespace (saes.statistics.*)."""

    @staticmethod
    def ttest_ind(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Student's two-sample independent t-test."""
        return IndependentTTest().run(groups, alpha=alpha)

    @staticmethod
    def ttest_welch(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Welch's two-sample independent t-test (unequal variances)."""
        return WelchTTest().run(groups, alpha=alpha)

    @staticmethod
    def ttest_rel(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Paired samples t-test."""
        return PairedTTest().run(groups, alpha=alpha)

    @staticmethod
    def anova_oneway(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Fisher One-Way ANOVA."""
        return OneWayANOVA().run(groups, alpha=alpha)

    @staticmethod
    def anova_welch(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Welch's One-Way ANOVA."""
        return WelchANOVA().run(groups, alpha=alpha)

    @staticmethod
    def shapiro(groups: dict[str, np.ndarray], alpha: float = 0.05) -> list[AssumptionResult]:
        """Check normality assumption via Shapiro-Wilk test."""
        return NormalityChecker(alpha=alpha).check(groups)

    @staticmethod
    def levene(groups: dict[str, np.ndarray], alpha: float = 0.05) -> AssumptionResult:
        """Check homogeneity of variance assumption via Levene's test."""
        return HomogeneityChecker(alpha=alpha).check(groups)


statistics = StatisticsNamespace()
