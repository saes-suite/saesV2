# Changelog

All notable changes to the SAES core repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Decoupled and extracted `chat`, `mcp`, and `plugins` into the dedicated `saes-ai` package.
- `saes.server.app` now dynamically mounts `saes_ai.server.routes` when `saes-ai` is installed without internal legacy fallbacks.
- Cleaned core dependencies by removing `chat` extra and delegating AI tooling to `[ai]` optional extra (`saes-ai`).

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
