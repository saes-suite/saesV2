"""Unit tests for SAES statistical chat agent."""

import io

import saes
from saes.chat.agent import StatisticalChatAgent
from saes.chat.context import build_analysis_context

SAMPLE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""


def test_build_analysis_context() -> None:
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    ctx = build_analysis_context(report, dataset)
    assert "ZDT1" in ctx
    assert "HV" in ctx
    assert "NSGAII" in ctx
    assert "MOEAD" in ctx


def test_statistical_chat_agent_offline_queries() -> None:
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    agent = StatisticalChatAgent(report=report, dataset=dataset)

    # Test "why" test selection query
    resp_why = agent.ask("Why was this statistical test chosen?")
    assert "Test Selection Rationale" in resp_why
    assert "ZDT1" in resp_why

    # Test "effect size" query
    resp_eff = agent.ask("What is the effect size?")
    assert "Effect Size" in resp_eff

    # Test "assumptions" query
    resp_assumptions = agent.ask("Were the normality and variance assumptions satisfied?")
    assert "Assumptions" in resp_assumptions

    # Test general summary
    resp_summary = agent.ask("Please summarize the findings.")
    assert "Statistical Findings Summary" in resp_summary


def test_statistical_chat_agent_providers_fallback() -> None:
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    # Local AI provider (offline fallback if Ollama not running)
    local_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="local",
        model="llama3.2",
        base_url="http://127.0.0.1:99999",
    )
    resp_local = local_agent.ask("Why Welch?")
    assert "Local AI error" in resp_local or "Test Selection Rationale" in resp_local

    # OpenAI missing key fallback
    openai_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="openai",
        api_key=None,
    )
    resp_openai = openai_agent.ask("Why Welch?")
    assert "OpenAI API key missing" in resp_openai

    # Anthropic missing key fallback
    anthropic_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="anthropic",
        api_key=None,
    )
    resp_anthropic = anthropic_agent.ask("Why Welch?")
    assert "Anthropic API key missing" in resp_anthropic

    # Gemini missing key fallback
    gemini_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="gemini",
        api_key=None,
    )
    resp_gemini = gemini_agent.ask("Why Welch?")
    assert "Google Gemini API key missing" in resp_gemini


def test_fetch_local_models_graceful_on_unreachable() -> None:
    models = StatisticalChatAgent.fetch_local_models(base_url="http://127.0.0.1:99999")
    assert models == []


def test_pull_ollama_model_empty_and_unreachable() -> None:
    # Empty model name
    res_empty = StatisticalChatAgent.pull_ollama_model(name="")
    assert res_empty["status"] == "error"

    # Unreachable base_url
    res_unreachable = StatisticalChatAgent.pull_ollama_model(name="llama3.2", base_url="http://127.0.0.1:99999")
    assert res_unreachable["status"] == "error"


def test_stream_pull_ollama_model_empty_and_unreachable() -> None:
    # Empty model name
    chunks_empty = list(StatisticalChatAgent.stream_pull_ollama_model(name=""))
    assert len(chunks_empty) == 1
    assert "error" in chunks_empty[0]

    # Unreachable base_url
    chunks_unreachable = list(
        StatisticalChatAgent.stream_pull_ollama_model(name="llama3.2", base_url="http://127.0.0.1:99999")
    )
    assert len(chunks_unreachable) == 1
    assert "error" in chunks_unreachable[0]


def test_statistical_chat_agent_stream_ask() -> None:
    dataset = saes.load(io.StringIO(SAMPLE_CSV))
    report = saes.analyze(dataset, generate_plots=False)

    # 1. Offline streaming
    agent = StatisticalChatAgent(report=report, dataset=dataset, provider="offline")
    tokens = list(agent.stream_ask("Why was this statistical test chosen?"))
    full_text = "".join(tokens)
    assert len(tokens) > 1
    assert "Test Selection Rationale" in full_text
    assert len(agent.history) == 2
    assert agent.history[-1]["content"] == full_text

    # 2. Local AI fallback streaming
    local_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="local",
        model="llama3.2",
        base_url="http://127.0.0.1:99999",
    )
    tokens_local = list(local_agent.stream_ask("Why Welch?"))
    full_local = "".join(tokens_local)
    assert "Local AI error" in full_local or "Test Selection Rationale" in full_local

    # 3. OpenAI missing key fallback streaming
    openai_agent = StatisticalChatAgent(
        report=report,
        dataset=dataset,
        provider="openai",
        api_key=None,
    )
    tokens_openai = list(openai_agent.stream_ask("Why Welch?"))
    full_openai = "".join(tokens_openai)
    assert "OpenAI API key missing" in full_openai


