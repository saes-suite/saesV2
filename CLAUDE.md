# CLAUDE.md — SAES (Stochastic Algorithm Evaluation Suite)

> This file is read automatically by Claude, Antigravity, and any AI assistant that opens this repository.
> It defines the project identity, architecture, and non-negotiable conventions.

## Project Identity

**What**: SAES (Stochastic Algorithm Evaluation Suite) — Modern Python scientific library and FastAPI REST server for statistical analysis, automated hypothesis testing, and benchmark evaluation.

**Who**: SAES development team.

**Goal**: Provide a clean, extensible statistical suite (t-tests, ANOVA, assumption checkers, decision engines, Pydantic domain models) and automated REST reporting for algorithm evaluation.

---

## Repository Layout

```
saes/
├── .agents/
│   └── rules/
│       └── code-guidelines.md   ← always_on agent context
├── .github/
│   └── workflows/
│       ├── tests.yml            ← CI pipeline (lint + test)
│       └── release.yml          ← Release stamp & bump on merged PRs
├── scripts/
│   └── release_stamp.js         ← Version & CHANGELOG automation
├── src/
│   └── saes/                    ← Core library source code
│       ├── api.py               ← High-level public API
│       ├── app/                 ← Execution pipeline
│       ├── domain/              ← Pydantic data models & exceptions
│       ├── io/                  ← CSV data readers & input handlers
│       ├── server/              ← FastAPI REST service (routes & schemas)
│       ├── stats/               ← Statistical algorithms (ANOVA, t-tests)
│       ├── validation/          ← Dataset validators
│       └── visualization/       ← Boxplot, violin, and plot generators
├── tests/                       ← Pytest suite (unit, statistical, integration)
├── CHANGELOG.md                 ← Release history
├── CLAUDE.md                    ← This file
├── Makefile                     ← All development commands
├── pyproject.toml               ← Project configuration & dependencies
├── README.md                    ← Project overview
├── uv.lock                      ← Dependency lockfile
└── version.txt                  ← Version (X.Y.Z)
```

---

## Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Language | Python >= 3.11 | Modern Python standard |
| Package Manager | `uv` | Fast package management & environment resolution |
| Data Validation | Pydantic V2 | Strict data modeling & canonical BenchmarkRecord |
| Statistical Engines | NumPy, SciPy, Pandas, Statsmodels | Core computations & dataframes |
| Visualization | Matplotlib, Seaborn, Plotly | Charting engines & PlotterRegistry |
| REST API | FastAPI, Uvicorn | High-performance async REST backend |
| Linter & Formatter | Ruff | Fast static analysis & formatting |
| Type Checker | MyPy | Strict type hints checking |
| Testing Framework | Pytest, Pytest-Cov | Unit, statistical correctness & integration tests |
| CI/CD | GitHub Actions | Workflows for test validation and release stamping |

---

## 🛡️ Mandatory Quality Gate

Before finishing any task or suggesting a commit, **you MUST run**:

```bash
make ci
```

This command runs the full quality pipeline (`lint` + `test`):
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest`

Never consider a task complete if `make ci` fails.

---

## 🌿 Branching & Versioning (Official Workflow)

### Version Tracking (`version.txt`)

- Tracked in `version.txt` and synchronized with `pyproject.toml`.
- Managed by `scripts/release_stamp.js` — **never edit manually**.
- Clean semver versions without `-SNAPSHOT` suffixes.

### Changelog Workflow

1. While working on a branch, add entries under `## [Unreleased]` in `CHANGELOG.md`.
   - Pending changes are tracked between `## [Unreleased]` and `## [Released]` markers.
2. On PR merge to `main`, the GitHub Actions `release.yml` workflow automatically:
   - Stamps `## [Unreleased]` into the `## [Released]` section with version, date, and author.
   - Keeps clean `## [Unreleased]` and `## [Released]` markers intact.
   - Bumps `version.txt` and `pyproject.toml` to the next clean semver release version.
   - Commits to `main` with `[skip ci]`.

### Branch → Version Bump Convention

| Branch prefix | Bump | Example |
|---|---|---|
| `feat/*` (default) | Minor | `feat/add-anova-viz` → `0.2.0` |
| `fix/*`, `patch/*`, `hotfix/*` | Patch | `fix/zero-variance-fallback` → `0.1.1` |
| `major/*` | Major | `major/v2-architecture` → `1.0.0` |

---

## Makefile Commands — Quick Reference

```bash
make help               # Show all targets
make install            # Install dependencies with uv (dev included)
make dev                # Run Streamlit GUI app
make start              # Run SAES CLI entry point
make build              # Build package wheel and sdist with uv
make lint               # Run ruff and mypy linters
make format             # Auto-format code with ruff
make test               # Run pytest test suite
make ci                 # Run full CI pipeline (lint + test)

make version            # Show current version from version.txt
make branch             # Interactive branch creation
make release-dry-run    # Preview changelog stamp
make release-stamp      # Stamp changelog + bump version
```

---

## Conventions for AI Agents

When working in this repository as an agent:

1. **Python Programmatic & REST API Parity**: Every feature built for `saes-ui` MUST first exist as a public Python API in `saes` (`src/saes/`) and be exposed via FastAPI (`src/saes/server/`). `saes-ui` never performs domain calculations itself—it consumes the `saes` API.
2. **Update `CHANGELOG.md`** under `## [Unreleased]` for any feature, fix, or breaking change.
3. **Never edit `version.txt` or `pyproject.toml` version manually** — use `make release-stamp`.
4. **Always run `make ci`** before completing a task to verify linting and tests.
5. **Follow Conventional Commits**: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`.
