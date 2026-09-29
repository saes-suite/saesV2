# Changelog

All notable changes to the SAES core repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Decoupled `AnalysisPipeline` from direct Matplotlib plotting imports.
- Created `saes.server` with FastAPI REST service endpoints (`/health`, `/api/v1/validate`, `/api/v1/analyze`, `/api/v1/chat`).
- Created `saes.chat` with `StatisticalChatAgent` supporting offline interpretation and Anthropic/OpenAI connectors.
- Prepared architectural structure for Model Context Protocol (MCP) in `saes.mcp`.
- Prepared Claude Plugin manifest and prompt instructions in `saes.plugins.claude`.
- Modularized dependencies into `[viz]`, `[api]`, `[chat]`, and `[all]` extras.

## [0.1.0] - 2026-08-19

### Added

- Initial release of SAES core statistics suite.
- One-way ANOVA and Welch's ANOVA tests.
- Two-sample independent, Welch's, and paired t-tests.
- Normality checking via Shapiro-Wilk test.
- Variance homogeneity checking via Levene's test.
- Automatic statistical test selection decision engine.
- Effect size calculations (Cohen's d, Eta-squared).
- High-level orchestration pipeline.
