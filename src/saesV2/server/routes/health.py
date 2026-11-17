"""Health check route."""

from fastapi import APIRouter

from saesV2.server.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse()
