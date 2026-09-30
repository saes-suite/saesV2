"""Statistical chat assistant REST endpoint."""

import io
from pathlib import Path

from fastapi import APIRouter

import saes
from saes.chat.agent import StatisticalChatAgent
from saes.server.schemas import ChatRequest, ChatResponse
from saes.server.storage import get_datasets_dir

router = APIRouter(prefix="/api/v1", tags=["Chat"])


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

    agent = StatisticalChatAgent(report=report, dataset=dataset)
    answer = agent.ask(req.query)
    return ChatResponse(response=answer, context_used=report is not None)
