"""Public API surface for SAES library."""

from typing import Any

import numpy as np

from saesV2.app.pipeline import AnalysisPipeline, AnalysisReport
from saesV2.domain.models import (
    AssumptionResult,
    BayesianRankResult,
    CriticalDistanceResult,
    DataExplorationResult,
    Dataset,
    PerformanceProfilesResult,
    PowerAnalysisResult,
    ReproducibilityResult,
    StatisticalResult,
    StatisticalTableResult,
    ValidationReport,
)
from saesV2.reports.reproducibility import (
    generate_environment_manifest,
    generate_jupyter_notebook,
    generate_python_script,
    generate_reproducibility_artifacts,
)
from saesV2.stats.anova import OneWayANOVA, WelchANOVA
from saesV2.stats.assumptions import HomogeneityChecker, NormalityChecker
from saesV2.stats.explorer import DataExplorerEngine, export_exploration_table
from saesV2.stats.nonparametric import (
    KruskalWallisTest,
    MannWhitneyUTest,
    WilcoxonSignedRankTest,
    adjust_pvalues,
    compute_cliffs_delta,
    compute_vargha_delaney_a12,
)
from saesV2.stats.power import (
    analyze_statistical_power,
    calculate_statistical_power,
    estimate_required_sample_size,
    generate_power_curves,
)
from saesV2.stats.profiles import compute_performance_profiles
from saesV2.stats.ranking import BayesianRankingTest, FriedmanNemenyiTest
from saesV2.stats.tables import StatisticalTableGenerator, export_statistical_table
from saesV2.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


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
    from saesV2.visualization.registry import PlotterRegistry

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


def generate_table(
    dataset: Dataset,
    metric: str | None = None,
    table_type: str = "instance_algorithm",
    test_type: str = "auto",
    posthoc_correction: str = "holm",
    summary_stat: str = "auto",
    reference_algorithm: str | None = None,
    scope: str = "all-instances",
    selected_instance: str | None = None,
    algorithms_filter: list[str] | None = None,
    instances_filter: list[str] | None = None,
    alpha: float = 0.05,
    precision: int = 4,
    latex_document_mode: str = "standalone",
    latex_notation: str = "scientific",
) -> StatisticalTableResult:
    """Generate a publication-grade ANOVA, Friedman, Wilcoxon, or Pairwise statistical benchmark table."""
    return StatisticalTableGenerator().generate(
        dataset=dataset,
        metric=metric,
        table_type=table_type,
        test_type=test_type,
        posthoc_correction=posthoc_correction,
        summary_stat=summary_stat,
        reference_algorithm=reference_algorithm,
        scope=scope,
        selected_instance=selected_instance,
        algorithms_filter=algorithms_filter,
        instances_filter=instances_filter,
        alpha=alpha,
        precision=precision,
        latex_document_mode=latex_document_mode,
        latex_notation=latex_notation,
    )


def export_table(
    table_result: StatisticalTableResult,
    export_format: str = "latex",
    document_mode: str | None = None,
    notation: str | None = None,
    title: str | None = None,
    caption: str | None = None,
    pdf_theme: str = "slate",
    pdf_orientation: str = "landscape",
    pdf_font_size: float = 8.5,
    pdf_highlight_best: bool = True,
    pdf_show_zebra: bool = True,
    pdf_grid_style: str = "full",
) -> bytes:
    """Export a generated StatisticalTableResult to LaTeX, CSV, Markdown, HTML, DOCX, or PDF bytes."""
    file_bytes, _, _ = export_statistical_table(
        table_result,
        export_format=export_format,
        document_mode=document_mode,
        notation=notation,
        title=title,
        caption=caption,
        pdf_theme=pdf_theme,
        pdf_orientation=pdf_orientation,
        pdf_font_size=pdf_font_size,
        pdf_highlight_best=pdf_highlight_best,
        pdf_show_zebra=pdf_show_zebra,
        pdf_grid_style=pdf_grid_style,
    )
    return file_bytes


def explore(
    dataset: Dataset,
    metrics: list[str] | None = None,
    algorithms: list[str] | None = None,
    instances: list[str] | None = None,
    alpha: float = 0.05,
) -> DataExplorationResult:
    """Explore dataset observations, reveal multi-metric tables, check normality, and get test recommendations."""
    return DataExplorerEngine().explore(
        dataset=dataset,
        metrics=metrics,
        algorithms=algorithms,
        instances=instances,
        alpha=alpha,
    )


def export_exploration(
    exploration: DataExplorationResult,
    view_mode: str = "reveal_metrics",
    export_format: str = "csv",
    stat_format: str = "mean_std",
    normality_filter: str = "all",
    output_path: str | None = None,
) -> bytes:
    """Export an Observations & Data exploration result into PDF, Word (.docx), LaTeX, CSV, Markdown, or HTML."""
    file_bytes, _, _ = export_exploration_table(
        exploration=exploration,
        view_mode=view_mode,
        export_format=export_format,
        stat_format=stat_format,
        normality_filter=normality_filter,
    )
    if output_path:
        with open(output_path, "wb") as f:
            f.write(file_bytes)
    return file_bytes


def profiles(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    max_tau: float | None = None,
    num_tau_points: int = 60,
) -> PerformanceProfilesResult:
    """Compute Dolan-Moré performance profiles and global Win/Tie/Loss meta-ranking."""
    return compute_performance_profiles(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        max_tau=max_tau,
        num_tau_points=num_tau_points,
    )


def power(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    target_effect_size: float = 0.50,
) -> PowerAnalysisResult:
    """Run full statistical power analysis, sample size planning, and post-hoc audit."""
    return analyze_statistical_power(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        target_power=target_power,
        target_effect_size=target_effect_size,
    )


def reproducibility(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    filename: str = "dataset.csv",
    raw_csv: str | None = None,
) -> ReproducibilityResult:
    """Generate standalone reproducibility code artifacts (Python script, Jupyter Notebook, manifest)."""
    return generate_reproducibility_artifacts(
        dataset=dataset,
        metric=metric,
        alpha=alpha,
        filename=filename,
        raw_csv=raw_csv,
    )


# Low-level programmatic statistics sub-namespace
class StatisticsNamespace:
    """Programmatic statistical test API namespace (saesV2.statistics.*)."""

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
    def wilcoxon(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Wilcoxon Signed-Rank paired non-parametric test."""
        return WilcoxonSignedRankTest().run(groups, alpha=alpha)

    @staticmethod
    def mann_whitney(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Mann-Whitney U (Wilcoxon Rank-Sum) independent non-parametric test."""
        return MannWhitneyUTest().run(groups, alpha=alpha)

    @staticmethod
    def kruskal(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Kruskal-Wallis H multi-group non-parametric omnibus test."""
        return KruskalWallisTest().run(groups, alpha=alpha)

    @staticmethod
    def anova_oneway(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Fisher One-Way ANOVA."""
        return OneWayANOVA().run(groups, alpha=alpha)

    @staticmethod
    def anova_welch(groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Welch's One-Way ANOVA."""
        return WelchANOVA().run(groups, alpha=alpha)

    @staticmethod
    def vargha_delaney_a12(x1: np.ndarray, x2: np.ndarray, maximize: bool = True) -> tuple[float, str]:
        """Compute Vargha-Delaney A12 non-parametric effect size and magnitude."""
        return compute_vargha_delaney_a12(x1, x2, maximize=maximize)

    @staticmethod
    def cliffs_delta(x1: np.ndarray, x2: np.ndarray, maximize: bool = True) -> tuple[float, str]:
        """Compute Cliff's delta non-parametric effect size and magnitude."""
        return compute_cliffs_delta(x1, x2, maximize=maximize)

    @staticmethod
    def adjust_pvalues(p_values: list[float], method: str = "holm") -> list[float]:
        """Adjust p-values for multiple comparisons (Holm, Bonferroni, Hochberg, FDR-BH)."""
        return adjust_pvalues(p_values, method=method)

    @staticmethod
    def table(
        dataset: Dataset,
        metric: str | None = None,
        table_type: str = "instance_algorithm",
        test_type: str = "auto",
        posthoc_correction: str = "holm",
        summary_stat: str = "auto",
        reference_algorithm: str | None = None,
        scope: str = "all-instances",
        selected_instance: str | None = None,
        algorithms_filter: list[str] | None = None,
        instances_filter: list[str] | None = None,
        alpha: float = 0.05,
        precision: int = 4,
    ) -> StatisticalTableResult:
        """Generate a statistical benchmark table (instance_algorithm, pairwise_matrix, omnibus_posthoc)."""
        return generate_table(
            dataset=dataset,
            metric=metric,
            table_type=table_type,
            test_type=test_type,
            posthoc_correction=posthoc_correction,
            summary_stat=summary_stat,
            reference_algorithm=reference_algorithm,
            scope=scope,
            selected_instance=selected_instance,
            algorithms_filter=algorithms_filter,
            instances_filter=instances_filter,
            alpha=alpha,
            precision=precision,
        )

    @staticmethod
    def explore(
        dataset: Dataset,
        metrics: list[str] | None = None,
        algorithms: list[str] | None = None,
        instances: list[str] | None = None,
        alpha: float = 0.05,
    ) -> DataExplorationResult:
        """Explore dataset distributions, reveal multi-metric pivots, and recommend statistical tests."""
        return explore(
            dataset=dataset,
            metrics=metrics,
            algorithms=algorithms,
            instances=instances,
            alpha=alpha,
        )

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

__all__ = [
    "load",
    "validate",
    "analyze",
    "plot",
    "generate_table",
    "export_table",
    "explore",
    "export_exploration",
    "profiles",
    "power",
    "reproducibility",
    "compute_performance_profiles",
    "analyze_statistical_power",
    "calculate_statistical_power",
    "estimate_required_sample_size",
    "generate_power_curves",
    "generate_reproducibility_artifacts",
    "generate_python_script",
    "generate_jupyter_notebook",
    "generate_environment_manifest",
    "statistics",
]

