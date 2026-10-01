"""Statistical chat assistant REST endpoint."""

import io
import json
from collections.abc import Generator
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import saes
from saes.chat.agent import StatisticalChatAgent
from saes.server.schemas import (
    ChatRequest,
    ChatResponse,
    LocalModelsResponse,
    OllamaPullRequest,
    OllamaPullResponse,
    ProviderProbeRequest,
    ProviderProbeResponse,
)
from saes.server.storage import get_datasets_dir

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.get("/chat/models/local", response_model=LocalModelsResponse)
def get_local_models(base_url: str = "http://localhost:11434") -> LocalModelsResponse:
    """Fetch installed models from a local Ollama or compatible server."""
    models = StatisticalChatAgent.fetch_local_models(base_url=base_url)
    return LocalModelsResponse(models=models, status="ok" if models else "empty")


@router.post("/chat/probe", response_model=ProviderProbeResponse)
def probe_provider_connection(req: ProviderProbeRequest) -> ProviderProbeResponse:
    """Test connectivity to any AI provider and retrieve available models."""
    res = StatisticalChatAgent.probe_provider(
        provider=req.provider,
        api_key=req.api_key,
        base_url=req.base_url,
    )
    return ProviderProbeResponse(**res)


@router.post("/chat/ollama/start")
def start_ollama_daemon() -> dict[str, bool]:
    """Attempt to start the local Ollama background server daemon."""
    started = StatisticalChatAgent.start_ollama_service()
    return {"started": started}


@router.post("/chat/ollama/pull", response_model=OllamaPullResponse)
def pull_ollama_model(req: OllamaPullRequest) -> OllamaPullResponse:
    """Download / pull a model into the local Ollama server."""
    res = StatisticalChatAgent.pull_ollama_model(
        name=req.name,
        base_url=req.base_url or "http://localhost:11434",
    )
    return OllamaPullResponse(**res)


@router.post("/chat/ollama/pull/stream")
def pull_ollama_model_stream(req: OllamaPullRequest) -> StreamingResponse:
    """Download / pull a model into the local Ollama server with progress streaming."""
    generator = StatisticalChatAgent.stream_pull_ollama_model(
        name=req.name,
        base_url=req.base_url or "http://localhost:11434",
    )
    return StreamingResponse(generator, media_type="application/x-ndjson")


@router.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest) -> ChatResponse:
    """Ask a question to the statistical AI agent."""
    dataset = None
    report = None

    # 1. Prefer loading from server-persisted dataset path if dataset_name is specified
    if req.dataset_name:
        try:
            safe_name = Path(req.dataset_name).name
            file_path = get_datasets_dir() / safe_name
            if file_path.exists() and file_path.is_file():
                dataset = saes.load(file_path)
                val = saes.validate(dataset)
                if val.is_valid:
                    report = saes.analyze(dataset, alpha=req.alpha, generate_plots=False)
        except Exception:
            pass

    # 2. Fallback to raw csv_data string if provided and dataset was not loaded from path
    if dataset is None and req.csv_data:
        try:
            dataset = saes.load(io.StringIO(req.csv_data))
            val = saes.validate(dataset)
            if val.is_valid:
                report = saes.analyze(dataset, alpha=req.alpha, generate_plots=False)
        except Exception:
            pass

    agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
    )
    answer = agent.ask(req.query)
    return ChatResponse(response=answer, context_used=report is not None)


@router.post("/chat/stream")
def chat_with_agent_stream(req: ChatRequest) -> StreamingResponse:
    """Ask a question to the statistical AI agent with live token streaming."""
    dataset = None
    report = None

    if req.dataset_name:
        try:
            safe_name = Path(req.dataset_name).name
            file_path = get_datasets_dir() / safe_name
            if file_path.exists() and file_path.is_file():
                dataset = saes.load(file_path)
                val = saes.validate(dataset)
                if val.is_valid:
                    report = saes.analyze(dataset, alpha=req.alpha, generate_plots=False)
        except Exception:
            pass

    if dataset is None and req.csv_data:
        try:
            dataset = saes.load(io.StringIO(req.csv_data))
            val = saes.validate(dataset)
            if val.is_valid:
                report = saes.analyze(dataset, alpha=req.alpha, generate_plots=False)
        except Exception:
            pass

    agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
    )

    def event_generator() -> Generator[str, None, None]:
        context_used = report is not None
        try:
            for token in agent.stream_ask(req.query):
                yield json.dumps({"token": token, "done": False, "context_used": context_used}) + "\n"
            yield json.dumps({"token": "", "done": True, "context_used": context_used}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e), "token": "", "done": True, "context_used": context_used}) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
