"""Health check route."""

from fastapi import APIRouter

from saes.server.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse()
