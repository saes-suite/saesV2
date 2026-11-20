"""FastAPI application factory for SAES."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saesV2._version import __version__
from saesV2.server.routes.analysis import router as analysis_router
from saesV2.server.routes.datasets import router as datasets_router
from saesV2.server.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="SAES REST API",
        description="Stochastic Algorithm Evaluation Suite - Automated Statistical Analysis & AI Workbench",
        version=__version__,
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

    return app


app = create_app()


def main() -> None:
    """CLI entry point for running the SAES REST API server via saes-api."""
    import uvicorn

    uvicorn.run("saesV2.server.app:app", host="0.0.0.0", port=8060, reload=True)


if __name__ == "__main__":
    main()
