"""Domain data models and immutable result value objects for SAES."""

from dataclasses import dataclass, field
from enum import Enum, auto

import pandas as pd


class Severity(Enum):
    """Validation issue severity level."""

    ERROR = auto()
    WARNING = auto()


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single structural, schema, or value validation issue."""

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
        return f"[{self.severity.name}] {self.issue_type}{loc}: {self.message}"


@dataclass
class ValidationReport:
    """Summary report of dataset validation results."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

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


@dataclass(frozen=True)
class AssumptionResult:
    """Outcome of a statistical assumption test (e.g. Shapiro-Wilk, Levene)."""

    name: str
    statistic: float
    p_value: float
    passed: bool
    alpha: float = 0.05
    details: str = ""


@dataclass(frozen=True)
class EffectSize:
    """Standardized effect size measure."""

    name: str  # e.g., "Cohen's d", "Hedges' g", "Eta-squared"
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    magnitude: str = ""  # e.g., "negligible", "small", "medium", "large"


@dataclass(frozen=True)
class StatisticalResult:
    """Immutable structured result of a statistical analysis test."""

    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: float | tuple[float, ...] | None
    alpha: float
    is_significant: bool
    effect_size: EffectSize | None
    confidence_interval: tuple[float, float] | None
    assumptions: tuple[AssumptionResult, ...]
    warnings: tuple[str, ...]
    interpretation: str
    group_summaries: dict[str, dict[str, float]]


@dataclass
class Dataset:
    """Represents a validated SAES dataset wrapping standard algorithm evaluation records."""

    df: pd.DataFrame

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
