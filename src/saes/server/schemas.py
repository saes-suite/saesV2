"""Pydantic request and response schemas for SAES REST API."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    service: str = "saes-api"


class ValidationIssueSchema(BaseModel):
    """Schema for individual validation issue."""

    severity: str
    issue_type: str
    message: str
    column: str | None = None
    row_index: int | None = None


class ValidationResponse(BaseModel):
    """Validation response report."""

    is_valid: bool
    error_count: int
    warning_count: int
    issues: list[ValidationIssueSchema]


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
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Analysis report response."""

    cells_count: int
    results: list[CellAnalysisSchema]


class ChatRequest(BaseModel):
    """User query request for the statistical chat assistant."""

    query: str
    csv_data: str | None = None
    alpha: float = 0.05


class ChatResponse(BaseModel):
    """Response from statistical chat assistant."""

    response: str
    context_used: bool
