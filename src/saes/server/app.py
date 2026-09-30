"""FastAPI application factory for SAES."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saes.server.routes.analysis import router as analysis_router
from saes.server.routes.chat import router as chat_router
from saes.server.routes.datasets import router as datasets_router
from saes.server.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="SAES REST API",
        description="Stochastic Algorithm Evaluation Suite - Automated Statistical Analysis & AI Workbench",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(datasets_router)
    app.include_router(analysis_router)
    app.include_router(chat_router)

    return app


app = create_app()
