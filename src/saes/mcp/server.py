"""Model Context Protocol (MCP) server skeleton for SAES.

This module provides the prepared architectural structure to expose SAES as an
MCP server for Claude Desktop, Claude Code, Cursor, and other MCP-compliant agents.

Implementation is planned for the MCP integration phase.
"""

from typing import Any

from saes.mcp.tools import SAES_MCP_TOOLS


class SAESMCPServer:
    """Model Context Protocol server exposing SAES statistical routines to LLMs."""

    def __init__(self, name: str = "saes-mcp", version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self.tools = SAES_MCP_TOOLS

    def list_tools(self) -> list[dict[str, Any]]:
        """Return list of supported MCP tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters_schema,
            }
            for tool in self.tools
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool call on behalf of LLM.

        Args:
            tool_name: Name of tool to execute.
            arguments: Tool arguments.

        Raises:
            NotImplementedError: Until the MCP server implementation phase is completed.
        """
        raise NotImplementedError(f"MCP Tool '{tool_name}' execution is scheduled for the MCP implementation phase.")

    def run_stdio(self) -> None:
        """Run MCP server over standard input/output (stdio transport)."""
        raise NotImplementedError("MCP stdio transport will be implemented in the MCP feature phase using `mcp` SDK.")


def main() -> None:
    """CLI entry point for saes-mcp server."""
    server = SAESMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
