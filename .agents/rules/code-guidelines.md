---
trigger: always_on
---

# SAES — Stochastic Algorithm Evaluation Suite

> This file is read automatically by Antigravity on every session.
> It defines the project identity, architecture, and non-negotiable conventions.

## Project Identity

**What**: SAES (Stochastic Algorithm Evaluation Suite) — Modern Python library and FastAPI REST server for statistical analysis, hypothesis testing, and algorithm evaluation.

**Who**: SAES development team.

**Goal**: Provide a clean, extensible statistical suite (t-tests, ANOVA, assumption checkers, decision engines) and automated REST reporting for algorithm evaluation.

---

## Non-Negotiable Architecture Rules (API-First & Python Parity)

1. **Python Programmatic API First**: Every feature in the SAES ecosystem (and every feature used by `saes-ui`) MUST be implemented first as a programmatic Python API in `saes` so Python developers have full parity.
2. **REST API Exposure**: Every UI-facing capability must be exposed through typed FastAPI endpoints in `src/saes/server/` using Pydantic V2 schemas.
3. **Single Source of Computation**: `saes` is the sole authority for statistical calculations, validation, and dataset repairs/transformations. `saes-ui` must never compute statistics or domain logic locally; it must always call `saes` via the REST API.

---

## Repository Layout

```
SAES2/
├── .agents/
│   └── rules/
│       └── code-guidelines.md   ← This file (always_on)
├── .github/
│   └── workflows/
│       ├── tests.yml            ← CI pipeline (lint + test)
│       └── release.yml          ← Release stamp & bump on merged PRs
├── docs/                        ← Project documentation
├── scripts/
│   └── release_stamp.js         ← Version & CHANGELOG automation
├── src/
│   └── saes/                    ← Core library source code
├── tests/                       ← Pytest suite (unit, statistical, integration)
├── CHANGELOG.md                 ← Release history
├── CLAUDE.md                    ← Claude & AI assistant context
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
| Package Manager | `uv` | Package installer and env manager |
| Stat Computation | NumPy, SciPy, Pandas, Statsmodels | Statistical routines |
| Visualization | Matplotlib, Seaborn | Chart generation |
| GUI | Streamlit | Web application interface |
| Quality | Ruff, MyPy | Static checking & type hints |
| Testing | Pytest, Pytest-Cov | Unit & integration tests |
| CI/CD | GitHub Actions | Workflows |

---

## Versioning & Changelog

| File | Purpose |
|------|---------|
| `version.txt` | Single source of truth for version |
| `pyproject.toml` | Package version metadata (synced with version.txt) |
| `CHANGELOG.md` | Release history following Keep a Changelog |

- **Never edit `version.txt` or `pyproject.toml` version manually** — use `make release-stamp`.
- Clean versions are maintained without `-SNAPSHOT` suffixes.
- Changelog tracks pending entries between `## [Unreleased]` and `## [Released]` markers.
- On PR merge to `main` / `master`, GitHub Actions stamps the changelog and bumps the version automatically.

### Branch → Bump Convention

| Prefix | Bump | Example |
|--------|------|---------|
| `feat/*` (default) | Minor | `feat/add-anova-viz` → `0.2.0` |
| `fix/*`, `patch/*`, `hotfix/*` | Patch | `fix/zero-variance-fallback` → `0.1.1` |
| `major/*` | Major | `major/v2-architecture` → `1.0.0` |

---

## Makefile Commands

```bash
make help               # All targets
make install            # Install dependencies via uv (dev included)
make dev                # Run Streamlit GUI app
make ui                 # Run Streamlit GUI app
make start              # Run SAES CLI entry point
make build              # Build wheel & sdist via uv
make lint               # Run ruff and mypy linters
make format             # Auto-format code with ruff
make test               # Run pytest test suite
make ci                 # Run full CI pipeline (lint + test)
make version            # Show current version
make branch             # Create branch (interactive)
make release-dry-run    # Preview changelog stamp
make release-stamp      # Stamp changelog + bump version
```

---

## 🛡️ Mandatory Quality Gate

Before finishing any task or suggesting a commit, **you MUST run**:

```bash
make ci
```

This command runs the full quality pipeline (`lint` + `test`). Never consider a task complete if this command fails.

---

## Agent Checklist

Before finishing any task in this repo:

- [ ] Did you update `CHANGELOG.md` under `## [Unreleased]`?
- [ ] Did you run `make ci` to verify all linters and pytest suite pass?
- [ ] Did you run `make release-dry-run` to preview version bump if needed?
- [ ] Follow conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).
