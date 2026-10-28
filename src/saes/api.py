"""Public API surface for SAES library."""

from typing import Any

import numpy as np

from saes.app.pipeline import AnalysisPipeline, AnalysisReport
from saes.domain.models import (
    AssumptionResult,
    BayesianRankResult,
    CriticalDistanceResult,
    Dataset,
    StatisticalResult,
    ValidationReport,
)
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.ranking import BayesianRankingTest, FriedmanNemenyiTest
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


def plot(
    dataset: Dataset,
    plot_type: str = "boxplot",
    metric: str | None = None,
    instance: str | None = None,
    interactive: bool = False,
    **kwargs: Any,
) -> Any:
    """Generate a publication-grade figure directly from a SAES Dataset.

    Supported plot_type values:
        - 'boxplot': Quartile box plot with centered observations
        - 'violin': Kernel density + internal box plot
        - 'critical-distance': Demšar (2006) Friedman + Nemenyi CD diagram
        - 'bayesian-rank': Dirichlet Process posterior rank distribution & ROPE
        - 'barplot', 'ecdf', 'lineplot', 'scatter'

    Args:
        dataset: SAES Dataset object.
        plot_type: Key registered in PlotterRegistry.
        metric: Metric identifier (defaults to first metric in dataset).
        instance: Problem instance identifier (defaults to first instance for cell plots).
        interactive: If True, returns a Plotly Figure; otherwise returns a Matplotlib Figure.
        **kwargs: Additional styling parameters (paper_format='golden'|'single-col'|..., palette, alpha, etc.).

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure.
    """
    from saes.visualization.registry import PlotterRegistry

    target_metric = metric or (dataset.metrics[0] if dataset.metrics else "value")
    resolved_plot_type = plot_type.lower()
    if resolved_plot_type in {"bayesian-sign", "bayesian-signed-rank"}:
        kwargs = {
            **kwargs,
            "method": "sign" if resolved_plot_type == "bayesian-sign" else "signed-rank",
        }
        plotter = PlotterRegistry.get("bayesian-rank")
    else:
        plotter = PlotterRegistry.get(plot_type)

    is_ranking = resolved_plot_type in {"critical-distance", "bayesian-rank", "bayesian-sign", "bayesian-signed-rank"}
    scope = str(kwargs.get("scope", "all-instances"))

    if is_ranking and scope == "all-instances":
        df_sub = dataset.df[dataset.df["metric"] == target_metric].copy()
        target_inst = instance or ""
    else:
        target_inst = instance or (dataset.instances[0] if dataset.instances else "")
        df_sub = dataset.get_cell(target_inst, target_metric) if target_inst else dataset.df.copy()

    plot_kwargs = {
        "metric": target_metric,
        "instance": target_inst,
        **kwargs,
    }
    if interactive:
        return plotter.plot_interactive(df_sub, **plot_kwargs)
    return plotter.plot(df_sub, **plot_kwargs)


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

    @staticmethod
    def friedman_nemenyi(
        dataset: Dataset,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> CriticalDistanceResult:
        """Friedman non-parametric rank test with Nemenyi post-hoc Critical Distance (CD)."""
        return FriedmanNemenyiTest().run(df=dataset.df, metric=metric, alpha=alpha, scope=scope, instance=instance)

    @staticmethod
    def bayesian_ranking(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        num_samples: int = 1500,
        algo_a: str | None = None,
        algo_b: str | None = None,
        method: str = "signed-rank",
    ) -> BayesianRankResult:
        """Bayesian Signed-Rank / Sign-Test Simplex (ROPE) and Posterior Ranking test (Benavoli et al. 2017)."""
        return BayesianRankingTest().run(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            num_samples=num_samples,
            algo_a=algo_a,
            algo_b=algo_b,
            method=method,
        )

    @staticmethod
    def bayesian_sign(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        num_samples: int = 1500,
        algo_a: str | None = None,
        algo_b: str | None = None,
    ) -> BayesianRankResult:
        """Bayesian Sign Test Simplex (ROPE) (Benavoli et al. 2017)."""
        return BayesianRankingTest().run(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            num_samples=num_samples,
            algo_a=algo_a,
            algo_b=algo_b,
            method="sign",
        )

    @staticmethod
    def bayesian_signed_rank(
        dataset: Dataset,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        num_samples: int = 1500,
        algo_a: str | None = None,
        algo_b: str | None = None,
    ) -> BayesianRankResult:
        """Bayesian Signed-Rank Test Simplex (ROPE) (Benavoli et al. 2017)."""
        return BayesianRankingTest().run(
            df=dataset.df,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            num_samples=num_samples,
            algo_a=algo_a,
            algo_b=algo_b,
            method="signed-rank",
        )


statistics = StatisticsNamespace()
