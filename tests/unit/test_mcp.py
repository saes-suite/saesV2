"""Unit tests for SAES MCP skeleton."""

import pytest

from saes.mcp.server import SAESMCPServer
from saes.mcp.tools import SAES_MCP_TOOLS


def test_mcp_tools_spec() -> None:
    assert len(SAES_MCP_TOOLS) >= 4
    tool_names = [t.name for t in SAES_MCP_TOOLS]
    assert "validate_benchmark_csv" in tool_names
    assert "run_statistical_analysis" in tool_names
    assert "check_assumptions" in tool_names
    assert "generate_distribution_plot" in tool_names


def test_mcp_server_skeleton() -> None:
    server = SAESMCPServer()
    tools = server.list_tools()
    assert len(tools) == len(SAES_MCP_TOOLS)

    # Calling tool in stub mode raises NotImplementedError
    with pytest.raises(NotImplementedError):
        server.call_tool("validate_benchmark_csv", {})
