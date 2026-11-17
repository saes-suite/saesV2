"""Pydantic request and response schemas for SAES REST API."""

from typing import Any

from pydantic import BaseModel, Field

from saesV2.domain.models import (
    AssumptionResult,
    BayesianRankResult,
    CriticalDistanceResult,
    DataExplorationResult,
    PerformanceProfilesResult,
    PowerAnalysisResult,
    ReproducibilityResult,
    StatisticalTableResult,
    ValidationIssue,
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    service: str = "saes-api"


# Backwards-compatible alias for ValidationIssue
ValidationIssueSchema = ValidationIssue


class ValidationResponse(BaseModel):
    """Validation response report."""

    is_valid: bool
    error_count: int
    warning_count: int
    issues: list[ValidationIssue]


class CellAnalysisSchema(BaseModel):
    """Statistical summary for an experimental cell."""

    instance: str
    metric: str
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Any
    is_significant: bool
    effect_size_name: str | None = None
    effect_size_value: float | None = None
    effect_size_magnitude: str | None = None
    assumptions: list[AssumptionResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    interpretation: str = ""
    group_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Analysis report response."""

    cells_count: int
    results: list[CellAnalysisSchema]


class DatasetItemSchema(BaseModel):
    """Metadata schema for a saved benchmark dataset."""

    name: str
    filename: str
    path: str
    size_bytes: int
    row_count: int | None = None
    modified_at: float | None = None


class DatasetDetailSchema(BaseModel):
    """Detailed response for a dataset including its CSV contents."""

    name: str
    filename: str
    path: str
    size_bytes: int
    row_count: int
    content: str


class DatasetUpdateSchema(BaseModel):
    """Payload for updating or saving CSV dataset content."""

    content: str


class RankingAnalysisResponse(BaseModel):
    """Combined response containing Friedman + Nemenyi CD and Bayesian Simplex (Sign & Signed-Rank) tests."""

    critical_distance: CriticalDistanceResult
    bayesian_ranking: BayesianRankResult
    bayesian_sign: BayesianRankResult | None = None
    bayesian_signed_rank: BayesianRankResult | None = None


StatisticalTableResponse = StatisticalTableResult
DataExplorationResponse = DataExplorationResult
PerformanceProfilesResponse = PerformanceProfilesResult
PowerAnalysisResponse = PowerAnalysisResult
ReproducibilityResponse = ReproducibilityResult

