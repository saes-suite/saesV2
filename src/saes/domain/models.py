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
