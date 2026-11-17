"""Domain data models, canonical benchmark records, and immutable result objects for SAES."""

from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkRecord(BaseModel):
    """Canonical strict schema for a single benchmark evaluation observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: str = Field(min_length=1, description="Algorithm identifier")
    instance: str = Field(min_length=1, description="Benchmark instance or problem name")
    metric: str = Field(min_length=1, description="Evaluation metric name (e.g. HV, IGD, Runtime)")
    run_id: int = Field(ge=0, description="Run seed or execution repetition identifier (>= 0)")
    value: float = Field(allow_inf_nan=False, description="Finite numeric observation value")
    maximize: bool = Field(description="Optimization direction: True to maximize, False to minimize")


class Severity(StrEnum):
    """Validation issue severity level."""

    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationIssue(BaseModel):
    """Represents a single structural, schema, or value validation issue."""

    model_config = ConfigDict(frozen=True)

    severity: Severity
    issue_type: str
    message: str
    column: str | None = None
    row_index: int | None = None

    def __str__(self) -> str:
        loc = ""
        if self.column and self.row_index is not None:
            loc = f" [Row {self.row_index}, Col '{self.column}']"
        elif self.column:
            loc = f" [Col '{self.column}']"
        elif self.row_index is not None:
            loc = f" [Row {self.row_index}]"
        return f"[{self.severity.value}] {self.issue_type}{loc}: {self.message}"


class ValidationReport(BaseModel):
    """Summary report of dataset validation results."""

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return all error-level issues."""
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return all warning-level issues."""
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def add_error(self, issue_type: str, message: str, column: str | None = None, row_index: int | None = None) -> None:
        """Add an error issue to report."""
        self.is_valid = False
        self.issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                issue_type=issue_type,
                message=message,
                column=column,
                row_index=row_index,
            )
        )

    def add_warning(
        self,
        issue_type: str,
        message: str,
        column: str | None = None,
        row_index: int | None = None,
    ) -> None:
        """Add a warning issue to report."""
        self.issues.append(
            ValidationIssue(
                severity=Severity.WARNING,
                issue_type=issue_type,
                message=message,
                column=column,
                row_index=row_index,
            )
        )


class AssumptionResult(BaseModel):
    """Outcome of a statistical assumption test (e.g. Shapiro-Wilk, Levene)."""

    model_config = ConfigDict(frozen=True)

    name: str
    statistic: float
    p_value: float
    passed: bool
    alpha: float = 0.05
    details: str = ""


class EffectSize(BaseModel):
    """Standardized effect size measure."""

    model_config = ConfigDict(frozen=True)

    name: str  # e.g., "Cohen's d", "Hedges' g", "Eta-squared"
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    magnitude: str = ""  # e.g., "negligible", "small", "medium", "large"


class StatisticalResult(BaseModel):
    """Immutable structured result of a statistical analysis test."""

    model_config = ConfigDict(frozen=True)

    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: float | tuple[float, ...] | str | None = None
    alpha: float
    is_significant: bool
    effect_size: EffectSize | None = None
    confidence_interval: tuple[float, float] | None = None
    assumptions: tuple[AssumptionResult, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    interpretation: str = ""
    group_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)


class AlgorithmRankItem(BaseModel):
    """Average Friedman rank and mean performance for a single algorithm."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    avg_rank: float
    mean_value: float


class CriticalDistanceResult(BaseModel):
    """Result of Friedman omnibus test, Iman-Davenport F, and Nemenyi Critical Distance (CD)."""

    model_config = ConfigDict(frozen=True)

    metric: str
    scope: str
    scope_label: str
    metrics_list: list[str] = Field(default_factory=list)
    instances_list: list[str] = Field(default_factory=list)
    metrics_label: str = ""
    instances_label: str = ""
    params_label: str = ""
    alpha: float
    k: int
    n_blocks: int
    cd: float
    friedman_stat: float
    friedman_p_value: float
    iman_davenport_f: float
    iman_davenport_p_value: float
    maximize: bool
    ranks: list[AlgorithmRankItem] = Field(default_factory=list)
    cliques: list[tuple[float, float]] = Field(default_factory=list)


class BayesianPosteriorItem(BaseModel):
    """Posterior rank distribution and ROPE probabilities for a single algorithm."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    mean_rank: float
    median_rank: float
    ci_lower: float
    ci_upper: float
    q25: float
    q75: float
    prob_best: float
    prob_better_than_ref: float
    prob_rope_with_ref: float
    samples: list[float] = Field(default_factory=list)


class BayesianRankResult(BaseModel):
    """Result of Bayesian Non-Parametric Signed-Rank / Sign-Test Simplex (ROPE) & Posterior Ranking test."""

    model_config = ConfigDict(frozen=True)

    metric: str
    scope: str
    scope_label: str
    bayesian_method: str = "signed-rank"
    metrics_list: list[str] = Field(default_factory=list)
    instances_list: list[str] = Field(default_factory=list)
    metrics_label: str = ""
    instances_label: str = ""
    params_label: str = ""
    k: int
    n_blocks: int
    rope_percent: float
    reference_algo: str
    maximize: bool
    algo_a: str = ""
    algo_b: str = ""
    algorithms: list[str] = Field(default_factory=list)
    p_left: float = 0.0
    p_rope: float = 0.0
    p_right: float = 0.0
    label_left: str = ""
    label_rope: str = ""
    label_right: str = ""
    points_x: list[float] = Field(default_factory=list)
    points_y: list[float] = Field(default_factory=list)
    points_density: list[float] = Field(default_factory=list)
    points_p_left: list[float] = Field(default_factory=list)
    points_p_rope: list[float] = Field(default_factory=list)
    points_p_right: list[float] = Field(default_factory=list)
    triangle_x: list[float] = Field(default_factory=list)
    triangle_y: list[float] = Field(default_factory=list)
    boundary_lines: list[tuple[float, float, float, float]] = Field(default_factory=list)
    items: list[BayesianPosteriorItem] = Field(default_factory=list)


class BenchmarkTableCell(BaseModel):
    """Cell (instance, algorithm) in a publication-grade benchmark performance & significance table."""

    model_config = ConfigDict(frozen=True)

    instance: str
    algorithm: str
    n: int
    mean: float
    std: float
    median: float
    iqr: float
    min_val: float
    max_val: float
    avg_rank: float
    formatted_value: str
    is_best: bool = False
    is_second_best: bool = False
    is_reference: bool = False
    significance_symbol: str = ""
    pairwise_test_name: str = ""
    pairwise_statistic: float | None = None
    pairwise_p_value: float | None = None
    pairwise_p_adjusted: float | None = None
    effect_size_name: str | None = None
    effect_size_value: float | None = None
    effect_size_magnitude: str | None = None
    a12: float | None = None
    cliffs_delta: float | None = None
    cohens_d: float | None = None
    normality_passed: bool = True
    shapiro_stat: float | None = None
    shapiro_p_value: float | None = None
    skewness: float = 0.0
    kurtosis: float = 0.0
    raw_values: list[float] = Field(default_factory=list)


class BenchmarkTableInstanceRow(BaseModel):
    """Row representing a single benchmark instance across all evaluated algorithms."""

    model_config = ConfigDict(frozen=True)

    instance: str
    best_algorithm: str
    second_best_algorithm: str = ""
    omnibus_test_name: str
    omnibus_statistic: float
    omnibus_p_value: float
    omnibus_df: str = ""
    omnibus_significant: bool
    omnibus_effect_size_name: str = ""
    omnibus_effect_size_value: float = 0.0
    omnibus_effect_size_magnitude: str = ""
    levene_statistic: float = 0.0
    levene_p_value: float = 1.0
    levene_passed: bool = True
    normality_passed_count: int = 0
    total_groups: int = 0
    interpretation: str = ""
    cells: dict[str, BenchmarkTableCell] = Field(default_factory=dict)


class AlgorithmSummaryFooter(BaseModel):
    """Global algorithm ranking, Win/Tie/Loss (+ / ≈ / −) counts, and post-hoc summary across instances."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    avg_rank: float
    rank_position: int
    mean_across_instances: float
    median_across_instances: float
    wins: int = 0
    ties: int = 0
    losses: int = 0
    best_count: int = 0
    second_best_count: int = 0
    friedman_z_vs_ref: float | None = None
    friedman_p_unadjusted: float | None = None
    friedman_p_holm: float | None = None
    friedman_p_bonferroni: float | None = None
    friedman_p_hochberg: float | None = None
    global_wilcoxon_stat: float | None = None
    global_wilcoxon_p: float | None = None
    global_wilcoxon_p_holm: float | None = None
    bayesian_prob_best: float | None = None
    bayesian_prob_better_than_ref: float | None = None
    bayesian_prob_rope_with_ref: float | None = None


class PairwiseInstanceComparison(BaseModel):
    """Per-instance comparison detail between a pair of algorithms (row vs col)."""

    model_config = ConfigDict(frozen=True)

    instance: str
    mean_row: float
    std_row: float
    median_row: float
    iqr_row: float
    mean_col: float
    std_col: float
    median_col: float
    iqr_col: float
    winner: str
    symbol: str = "="
    is_significant: bool = False
    p_value: float | None = None
    a12: float | None = None


class PairwiseMatrixCell(BaseModel):
    """Cell (algo_row, algo_col) in a k x k pairwise statistical comparison matrix."""

    model_config = ConfigDict(frozen=True)

    algo_row: str
    algo_col: str
    test_name: str
    statistic: float
    p_value: float
    p_adjusted: float
    is_significant: bool
    direction: str
    symbol: str
    instance_symbols: str = ""
    sig_wins: int = 0
    sig_ties: int = 0
    sig_losses: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    mean_row: float = 0.0
    mean_col: float = 0.0
    median_row: float = 0.0
    median_col: float = 0.0
    effect_size_name: str = ""
    effect_size_value: float = 0.0
    effect_size_magnitude: str = ""
    a12: float = 0.5
    cliffs_delta: float = 0.0
    cohens_d: float = 0.0
    prob_row_better: float | None = None
    prob_rope: float | None = None
    prob_col_better: float | None = None
    interpretation: str = ""
    instance_breakdown: list[PairwiseInstanceComparison] = Field(default_factory=list)


class StatisticalTableResult(BaseModel):
    """Complete result of a statistical benchmark table generation (ANOVA, Friedman, Wilcoxon, etc.)."""

    model_config = ConfigDict(frozen=True)

    table_type: str
    metric: str
    maximize: bool
    alpha: float
    test_type: str
    resolved_omnibus_test: str
    resolved_pairwise_test: str
    posthoc_correction: str
    summary_stat: str
    scope: str
    selected_instance: str | None = None
    reference_algorithm: str
    algorithms: list[str] = Field(default_factory=list)
    instances: list[str] = Field(default_factory=list)
    friedman_stat: float | None = None
    friedman_p_value: float | None = None
    iman_davenport_f: float | None = None
    iman_davenport_p_value: float | None = None
    kendalls_w: float | None = None
    nemenyi_cd: float | None = None
    global_anova_f: float | None = None
    global_anova_p: float | None = None
    global_welch_f: float | None = None
    global_welch_p: float | None = None
    global_kruskal_h: float | None = None
    global_kruskal_p: float | None = None
    instance_rows: list[BenchmarkTableInstanceRow] = Field(default_factory=list)
    algorithm_summaries: list[AlgorithmSummaryFooter] = Field(default_factory=list)
    pairwise_matrix: list[list[PairwiseMatrixCell]] = Field(default_factory=list)
    latex_document_mode: str = "standalone"
    latex_notation: str = "scientific"
    latex_preview: str = ""
    markdown_preview: str = ""
    caption: str = ""


class DistributionProfile(BaseModel):
    """Detailed distribution, normality, and outlier diagnostics for an (algorithm, instance, metric) group."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    instance: str
    metric: str
    maximize: bool
    n: int
    mean: float
    std: float
    median: float
    iqr: float
    min_val: float
    max_val: float
    cv: float
    skewness: float
    kurtosis: float
    outlier_count: int
    shapiro_stat: float
    shapiro_p: float
    shapiro_passed: bool
    dagostino_stat: float | None = None
    dagostino_p: float | None = None
    dagostino_passed: bool | None = None
    ks_stat: float
    ks_p: float
    ks_passed: bool
    is_normal: bool
    distribution_shape: str
    quantiles: dict[str, float] = Field(default_factory=dict)
    raw_values: list[float] = Field(default_factory=list)


class PivotedObservationRow(BaseModel):
    """Observation row where all metrics for (algorithm, instance, run_id) are revealed side-by-side."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    instance: str
    run_id: int
    metrics_values: dict[str, float] = Field(default_factory=dict)


class AggregatedMetricRow(BaseModel):
    """Aggregated (algorithm, instance) row revealing all metrics (Mean±SD, Median(IQR), Normality) side-by-side."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    instance: str
    n_runs: int
    metrics_mean_std: dict[str, str] = Field(default_factory=dict)
    metrics_median_iqr: dict[str, str] = Field(default_factory=dict)
    metrics_normal: dict[str, bool] = Field(default_factory=dict)
    metrics_shapiro_p: dict[str, float] = Field(default_factory=dict)


class ContextDiagnostics(BaseModel):
    """Global experimental design and assumption diagnostics for the currently filtered dataset slice."""

    model_config = ConfigDict(frozen=True)

    k_algorithms: int
    n_instances: int
    n_metrics: int
    total_observations: int
    runs_per_cell: int
    is_paired: bool
    is_balanced: bool
    normality_pass_count: int
    normality_total_groups: int
    normality_rate: float
    all_normal: bool
    homoscedasticity_levene_stat: float
    homoscedasticity_levene_p: float
    homoscedasticity_passed: bool
    homoscedasticity_bartlett_stat: float
    homoscedasticity_bartlett_p: float
    total_outliers: int


class TestRecommendation(BaseModel):
    """Context-aware statistical test recommendation with rationale and literature citation."""

    model_config = ConfigDict(frozen=True)

    test_id: str
    test_name: str
    category: str
    suitability: str
    suitability_score: int
    rationale: str
    when_to_use: str
    assumptions_satisfied: list[str] = Field(default_factory=list)
    assumptions_violated: list[str] = Field(default_factory=list)
    suggested_table_type: str
    posthoc_suggestion: str = ""
    effect_size_suggestion: str = ""
    reference_citation: str = ""


class DataExplorationResult(BaseModel):
    """Complete response for dataset exploration, multi-metric reveal, normality profiling, and test recommendations."""

    model_config = ConfigDict(frozen=True)

    available_metrics: list[str] = Field(default_factory=list)
    available_algorithms: list[str] = Field(default_factory=list)
    available_instances: list[str] = Field(default_factory=list)
    metric_directions: dict[str, bool] = Field(default_factory=dict)
    active_metrics: list[str] = Field(default_factory=list)
    active_algorithms: list[str] = Field(default_factory=list)
    active_instances: list[str] = Field(default_factory=list)
    alpha: float = 0.05
    context_diagnostics: ContextDiagnostics
    recommendations: list[TestRecommendation] = Field(default_factory=list)
    distribution_profiles: list[DistributionProfile] = Field(default_factory=list)
    pivoted_rows: list[PivotedObservationRow] = Field(default_factory=list)
    aggregated_rows: list[AggregatedMetricRow] = Field(default_factory=list)


class Dataset:
    """Represents a validated SAES dataset wrapping standard algorithm evaluation records."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def algorithms(self) -> list[str]:
        """List of unique algorithm names in dataset."""
        return sorted(self.df["algorithm"].unique().tolist())

    @property
    def instances(self) -> list[str]:
        """List of unique problem instances in dataset."""
        return sorted(self.df["instance"].unique().tolist())

    @property
    def metrics(self) -> list[str]:
        """List of unique metrics evaluated in dataset."""
        return sorted(self.df["metric"].unique().tolist())

    def get_cell(self, instance: str, metric: str) -> pd.DataFrame:
        """Filter dataset for a specific (instance, metric) experimental context."""
        return self.df[(self.df["instance"] == instance) & (self.df["metric"] == metric)].copy()

    def to_records(self) -> list[BenchmarkRecord]:
        """Convert underlying DataFrame observations into strictly validated BenchmarkRecord models."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[BenchmarkRecord])
        return adapter.validate_python(self.df.to_dict(orient="records"))



class AlgorithmProfileCurve(BaseModel):
    """Performance profile curve points for an individual algorithm."""

    model_config = ConfigDict(frozen=True)

    algorithm: str
    tau_values: list[float] = Field(default_factory=list)
    rho_values: list[float] = Field(default_factory=list)
    efficiency: float  # rho(1.0)
    robustness: float  # rho(tau_max)


class WinTieLossCell(BaseModel):
    """Cell in the global Win/Tie/Loss comparison matrix."""

    model_config = ConfigDict(frozen=True)

    algo_row: str
    algo_col: str
    wins: int
    ties: int
    losses: int
    net: int


class GlobalMetaRankingItem(BaseModel):
    """Aggregate benchmark meta-ranking item across all instances."""

    model_config = ConfigDict(frozen=True)

    rank: int
    algorithm: str
    efficiency: float
    robustness: float
    total_wins: int
    total_ties: int
    total_losses: int
    net_score: int
    copeland_score: float


class PerformanceProfilesResult(BaseModel):
    """Complete result of Dolan-Moré performance profiles and global meta-benchmarking."""

    model_config = ConfigDict(frozen=True)

    metric: str
    maximize: bool
    alpha: float
    tau_values: list[float] = Field(default_factory=list)
    curves: list[AlgorithmProfileCurve] = Field(default_factory=list)
    wtl_matrix: list[list[WinTieLossCell]] = Field(default_factory=list)
    meta_rankings: list[GlobalMetaRankingItem] = Field(default_factory=list)


class PowerCurveSeries(BaseModel):
    """Sample size vs power curve series for a specific effect size."""

    model_config = ConfigDict(frozen=True)

    effect_size_label: str
    effect_size_value: float
    sample_sizes: list[int] = Field(default_factory=list)
    power_values: list[float] = Field(default_factory=list)


class SampleSizeRecommendation(BaseModel):
    """A priori sample size recommendation for a target power and effect size."""

    model_config = ConfigDict(frozen=True)

    target_power: float
    alpha: float
    effect_size: float
    effect_size_label: str
    recommended_n_independent: int
    recommended_n_paired: int
    recommended_n_nonparametric: int


class PostHocPowerItem(BaseModel):
    """Post-hoc statistical power evaluation for a specific algorithm comparison."""

    model_config = ConfigDict(frozen=True)

    algo_1: str
    algo_2: str
    instance: str
    metric: str
    n_samples: int
    cohens_d: float
    a12: float
    achieved_power: float
    power_status: str  # Optimal (>=0.8), Adequate (0.7-0.8), Underpowered (<0.7)
    is_significant: bool
    recommendation: str


class PowerAnalysisResult(BaseModel):
    """Complete statistical power analysis and experimental sample size planning."""

    model_config = ConfigDict(frozen=True)

    alpha: float
    target_power: float
    target_effect_size: float
    sample_size_recommendations: list[SampleSizeRecommendation] = Field(default_factory=list)
    power_curves: list[PowerCurveSeries] = Field(default_factory=list)
    post_hoc_audit: list[PostHocPowerItem] = Field(default_factory=list)
    overall_power_health: str


class EnvironmentManifest(BaseModel):
    """Reproducibility metadata and execution environment manifest."""

    model_config = ConfigDict(frozen=True)

    python_version: str
    platform: str
    saes_version: str
    scipy_version: str
    numpy_version: str
    pandas_version: str
    dataset_checksum_sha256: str
    timestamp: str
    row_count: int
    runs_per_cell: int


class ReproducibilityResult(BaseModel):
    """Generated standalone reproducible code artifacts and environment specification."""

    model_config = ConfigDict(frozen=True)

    python_script: str
    jupyter_notebook_json: str
    manifest: EnvironmentManifest

