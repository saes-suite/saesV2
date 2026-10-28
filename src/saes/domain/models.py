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
