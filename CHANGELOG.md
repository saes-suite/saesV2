# Changelog

All notable changes to the SAES core repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Released]

## [0.2.0] — 2026-11-18 (@rorro6787)

### Added

- Initial release of SAES core statistics suite.
- One-way ANOVA and Welch's ANOVA tests.
- Two-sample independent, Welch's, and paired t-tests.
- Normality checking via Shapiro-Wilk test.
- Variance homogeneity checking via Levene's test.
- Automatic statistical test selection decision engine.
- Effect size calculations (Cohen's d, Eta-squared).
- High-level orchestration pipeline.
- `saesV2.export_plot(...)` programmatic Python API to export any registered statistical plot or diagram (`boxplot`, `violin`, `critical-distance`, `bayesian-rank`, `barplot`, `lineplot`, `ecdf`, `scatter`) as PNG (300 DPI), SVG, or PDF bytes.
- REST API plot export endpoints `POST /api/v1/plots/export` and `POST /api/v1/datasets/{filename}/plots/export` returning downloadable PNG, SVG, or PDF figures.
- Dolan-Moré (2002) performance profiles and meta-benchmarking engine (`saes.stats.profiles.compute_performance_profiles`, `saes.profiles`) computing solver efficiency $\rho(1)$, robustness $\lim \rho(\tau)$, $\tau$ curves, and global pairwise Win/Tie/Loss (W/T/L) and Copeland score ranking matrices across multi-problem benchmark suites.
- Statistical power and experimental design planning engine (`saes.stats.power.analyze_statistical_power`, `calculate_statistical_power`, `estimate_required_sample_size`, `generate_power_curves`, `saes.power`) implementing exact two-sample power via non-central Student $t$ distribution (`scipy.stats.nct`), Hodges-Lehmann non-parametric sample size bounds, and post-hoc power audits to prevent Type II errors in stochastic algorithm evaluation.
- FAIR science reproducibility pack generator (`saes.reports.reproducibility.generate_reproducibility_artifacts`, `saes.reproducibility`) creating standalone executable Python scripts (`reproduce.py` with `import saes`), Jupyter Notebooks (`.ipynb` nbformat 4), and environment SHA-256 integrity manifests.
- REST API endpoints for specialized scientific capabilities: `POST /api/v1/profiles`, `POST /api/v1/power`, `POST /api/v1/reproducibility`, and dataset-scoped counterparts `/api/v1/datasets/{filename}/profiles`, `/api/v1/datasets/{filename}/power`, `/api/v1/datasets/{filename}/reproducibility`.
- Publication-grade PDF styling & presentation customization in `_build_pdf_bytes`, `export_statistical_table`, and `saes.export_table` with support for custom title, custom caption, 5 distinct publication color themes (`slate`, `academic_bw`, `emerald`, `navy`, `crimson`), orientation (`landscape` vs `portrait`), font density scaling (`compact` 7pt, `standard` 8.5pt, `large` 10pt), best performer winner highlighting, zebra striping, and border rule styles (`full`, `horizontal` booktabs, `minimal`).
- REST API support for PDF table presentation options across `POST /api/v1/tables/export`, `GET /api/v1/datasets/{filename}/tables/export`, and `POST /api/v1/datasets/{filename}/tables/export`.
- Standalone LaTeX document generation (`latex_document_mode`: `standalone` vs `snippet`) with complete LaTeX document preambles and postambles (`\documentclass`, `\usepackage[table]{xcolor}`, `booktabs`, `amsmath`, `\begin{document}`, `\maketitle`) allowing immediate out-of-the-box compilation in Overleaf or TeXShop.
- Configurable LaTeX number notation modes (`latex_notation`: `scientific` `$2.55\text{e-}02_{...}$`, `siunitx` `\SI{2.55e-2}{...}`, and `standard` `$0.0255 \pm 0.0033$`) with cell shading (`\cellcolor{gray95}` for best, `\cellcolor{gray25}` for second best), adaptive `\tabcolsep` column spacing, and vertical grid borders matching standard scientific benchmark styles.
- Non-parametric hypothesis tests (`WilcoxonSignedRankTest`, `MannWhitneyUTest`, `KruskalWallisTest`), non-parametric effect sizes (`compute_vargha_delaney_a12`, `compute_cliffs_delta`, Epsilon-squared), and multiple-testing p-value corrections (`adjust_pvalues`: Holm-Bonferroni, Bonferroni-Dunn, Hochberg, Benjamini-Hochberg FDR) in `saes.stats.nonparametric` and `saes.statistics.*`.
- Scientific benchmark table generation (`StatisticalTableGenerator`, `saes.generate_table`, `saes.statistics.table`) and multi-format table exporter (`export_statistical_table`, `saes.export_table`) supporting Instance × Algorithm (+ / ≈ / −) benchmark tables, 1vs1 pairwise comparison matrices with per-instance statistical confidence symbol sequences (`+ / = / -`, `instance_symbols`, standalone LaTeX `1vs1 Table` with `\texttt{...}` and ordered instance caption), Friedman/ANOVA post-hoc control tables, and exports in PDF, Word (`.docx`), LaTeX, CSV, Markdown, and HTML.
- Dataset observation explorer, multi-metric side-by-side pivot reveal, distribution/normality profiler (Shapiro-Wilk, D'Agostino-Pearson $K^2$, Kolmogorov-Smirnov, Levene, Bartlett, IQR outliers), context-aware statistical test recommendation engine (`DataExplorerEngine`, `saes.explore`, `saes.statistics.explore`), and multi-format observation table exporter (`export_exploration_table`, `saes.export_exploration`).
- REST API endpoints `POST /api/v1/tables`, `POST /api/v1/tables/export`, `POST /api/v1/explore`, `POST /api/v1/explore/export`, and HTTP-cacheable dataset-scoped `GET`/`POST` counterparts `/api/v1/datasets/{filename}/tables`, `/api/v1/datasets/{filename}/tables/export`, `/api/v1/datasets/{filename}/explore`, and `/api/v1/datasets/{filename}/explore/export`.
- Non-parametric Friedman rank test with Iman-Davenport F and Nemenyi post-hoc Critical Distance (`FriedmanNemenyiTest`, `saes.statistics.friedman_nemenyi`), plus Bayesian Sign and Signed-Rank Simplex ROPE & posterior ranking tests (`BayesianRankingTest`, `saes.statistics.bayesian_ranking`, `bayesian_sign`, `bayesian_signed_rank`) with multi-metric (`all-metrics`) support.
- Publication-grade `CriticalDistancePlotGenerator` (`critical-distance`) and `BayesianRankPlotGenerator` (`bayesian-rank`, `bayesian-sign`, `bayesian-signed-rank`) supporting Matplotlib (`plot`) and Plotly (`plot_interactive`), registered in `PlotterRegistry` and exposed via `saes.plot(...)`.
- Scientific paper aspect ratio presets (`PAPER_FORMAT_PRESETS`: `single-col`, `golden`, `double-col`, `square`, `compact`) and palettes (`academic`, `nature`, `viridis`, `colorblind`, `set2`) in `saes.visualization.utils`.
- REST API endpoints `POST /api/v1/ranking` and `POST /api/v1/datasets/{filename}/ranking`, plus `interpretation` and `group_summaries` in `/api/v1/analyze` responses.
- PEP 561 `py.typed` marker, core `pydantic>=2.0.0` dependency, and lazy loading for `PlotterRegistry`.

### Fixed

- Fixed LaTeX table compilation errors in `render_latex_table` and `export_flat_table` where unescaped Unicode characters (`±`, `α`, `−`, `≈`, `—`, `²`) in benchmark table captions, cells, and headers caused fatal `pdflatex` compilation errors (`Unicode character not set up for use with LaTeX`). Added automated conversion of mathematical symbols to standard LaTeX math commands and `\footnotesize` scaling for publication compliance.

### Changed

- Decoupled and extracted `chat`, `mcp`, and `plugins` into the dedicated `saes-ai` package.
- `saes.server.app` now dynamically mounts `saes_ai.server.routes` when `saes-ai` is installed without internal legacy fallbacks.
- Cleaned core dependencies by removing `chat` extra and delegating AI tooling to `[ai]` optional extra (`saes-ai`).

