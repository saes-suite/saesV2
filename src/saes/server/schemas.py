"""Pydantic request and response schemas for SAES REST API."""

from typing import Any

from pydantic import BaseModel, Field

from saes.domain.models import AssumptionResult, ValidationIssue


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


class ChatRequest(BaseModel):
    """User query request for the statistical chat assistant."""

    query: str
    dataset_name: str | None = None
    csv_data: str | None = None
    alpha: float = 0.05
    provider: str = "offline"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class ChatResponse(BaseModel):
    """Response from statistical chat assistant."""

    response: str
    context_used: bool


class LocalModelsResponse(BaseModel):
    """List of models available in local Ollama or compatible server."""

    models: list[str] = Field(default_factory=list)
    status: str = "ok"


class ProviderProbeRequest(BaseModel):
    """Request to probe AI provider connectivity and retrieve available models."""

    provider: str
    api_key: str | None = None
    base_url: str | None = None


class ProviderProbeResponse(BaseModel):
    """Response of provider connectivity probe with dynamic models list."""

    status: str = "ok"
    models: list[str] = Field(default_factory=list)
    message: str | None = None
    can_start_ollama: bool = False
    binary_path: str | None = None


class OllamaPullRequest(BaseModel):
    """Request to download a model into local Ollama daemon."""

    name: str
    base_url: str | None = None


class OllamaPullResponse(BaseModel):
    """Response of Ollama pull operation."""

    status: str = "ok"
    message: str
