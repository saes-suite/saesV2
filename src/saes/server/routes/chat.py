"""Statistical chat assistant REST endpoint."""

import io

from fastapi import APIRouter

import saes
from saes.chat.agent import StatisticalChatAgent
from saes.server.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest) -> ChatResponse:
    """Ask a question to the statistical AI agent."""
    dataset = None
    report = None

    if req.csv_data:
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
