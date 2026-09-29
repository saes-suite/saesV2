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
