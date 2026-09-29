"""Tool definitions and schemas for the SAES Model Context Protocol (MCP) server.

NOTE: This is the prepared structural architecture for MCP integration.
Implementation is stubbed until the MCP server feature phase is activated.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPToolSpec:
    """Specification of an MCP tool exposed to LLMs."""

    name: str
    description: str
    parameters_schema: dict[str, Any]


SAES_MCP_TOOLS: list[MCPToolSpec] = [
    MCPToolSpec(
        name="validate_benchmark_csv",
        description="Validate the schema, column names, data types, and value constraints of an algorithm benchmark CSV file.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the CSV benchmark file on disk.",
                },
            },
            "required": ["file_path"],
        },
    ),
    MCPToolSpec(
        name="run_statistical_analysis",
        description="Run automated statistical evaluation (normality, variance homogeneity, t-tests or ANOVA, effect sizes) on a benchmark CSV.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to benchmark CSV.",
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level (default: 0.05).",
                    "default": 0.05,
                },
                "instance": {
                    "type": "string",
                    "description": "Optional specific problem instance to analyze (e.g. 'ZDT1').",
                },
                "metric": {
                    "type": "string",
                    "description": "Optional specific metric to analyze (e.g. 'HV').",
                },
            },
            "required": ["file_path"],
        },
    ),
    MCPToolSpec(
        name="check_assumptions",
        description="Test normality (Shapiro-Wilk) and homoscedasticity (Levene's test) for groups in a benchmark experiment.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "instance": {"type": "string", "description": "Problem instance identifier."},
                "metric": {"type": "string", "description": "Metric name."},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": ["file_path", "instance", "metric"],
        },
    ),
    MCPToolSpec(
        name="generate_distribution_plot",
        description="Generate statistical visualization (boxplot, violin, bar, ecdf) for benchmark results.",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to benchmark CSV."},
                "plot_type": {
                    "type": "string",
                    "enum": ["boxplot", "violin", "barplot", "ecdf", "scatterplot"],
                    "description": "Type of distribution plot to generate.",
                },
                "instance": {"type": "string", "description": "Problem instance identifier."},
                "metric": {"type": "string", "description": "Metric name."},
                "output_path": {"type": "string", "description": "File path to save the generated plot image."},
            },
            "required": ["file_path", "plot_type", "instance", "metric"],
        },
    ),
]
