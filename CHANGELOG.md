# Changelog

All notable changes to the SAES core repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Non-parametric Friedman rank test with Iman-Davenport F and Nemenyi post-hoc Critical Distance (`FriedmanNemenyiTest`, `saes.statistics.friedman_nemenyi`), plus Bayesian Sign and Signed-Rank Simplex ROPE & posterior ranking tests (`BayesianRankingTest`, `saes.statistics.bayesian_ranking`, `bayesian_sign`, `bayesian_signed_rank`) with multi-metric (`all-metrics`) support.
- Publication-grade `CriticalDistancePlotGenerator` (`critical-distance`) and `BayesianRankPlotGenerator` (`bayesian-rank`, `bayesian-sign`, `bayesian-signed-rank`) supporting Matplotlib (`plot`) and Plotly (`plot_interactive`), registered in `PlotterRegistry` and exposed via `saes.plot(...)`.
- Scientific paper aspect ratio presets (`PAPER_FORMAT_PRESETS`: `single-col`, `golden`, `double-col`, `square`, `compact`) and palettes (`academic`, `nature`, `viridis`, `colorblind`, `set2`) in `saes.visualization.utils`.
- REST API endpoints `POST /api/v1/ranking` and `POST /api/v1/datasets/{filename}/ranking`, plus `interpretation` and `group_summaries` in `/api/v1/analyze` responses.
- PEP 561 `py.typed` marker, core `pydantic>=2.0.0` dependency, and lazy loading for `PlotterRegistry`.

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
