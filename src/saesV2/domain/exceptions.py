"""Custom exception hierarchy for SAES library."""

from typing import Any


class SAESError(Exception):
    """Base exception for all SAES errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputError(SAESError):
    """Raised when input files or formats are invalid or unreadable."""


class ValidationError(SAESError):
    """Raised when dataset structure, schema, or types fail validation."""


class ConfigurationError(SAESError):
    """Raised when statistical analysis or engine configuration is invalid."""


class StatisticalAnalysisError(SAESError):
    """Raised when statistical calculation fails (e.g. zero variance, insufficient samples)."""


class VisualizationError(SAESError):
    """Raised when plot generation or visual rendering fails."""


class InfrastructureError(SAESError):
    """Raised when underlying system or I/O resources fail."""
