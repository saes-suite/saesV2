<p align="center">
  <img src="https://raw.githubusercontent.com/saes-suite/saesV2/main/assets/saes-logo.svg" alt="SAES Logo" width="128" />
</p>

# SAES — Stochastic Algorithm Evaluation Suite (`saesV2`)

Modern, high-performance Python library and API service for automated statistical analysis, hypothesis testing, and benchmark evaluation of stochastic algorithms.

---

## 📦 Installation

```bash
# 1. Core library (lightweight)
pip install saesV2

# 2. With visualization support (Matplotlib, Seaborn, Plotly)
pip install "saesV2[viz]"

# 3. With REST API service (FastAPI, Uvicorn)
pip install "saesV2[api]"

# 4. Full installation (viz + api)
pip install "saesV2[all]"
```

## 🚀 Quick Usage

### Python Programmatic API

```python
import saesV2

# 1. Load benchmark dataset
dataset = saesV2.load("data.csv")

# 2. Validate format & constraints
validation = saesV2.validate(dataset)
assert validation.is_valid

# 3. Automated statistical analysis & test selection
report = saesV2.analyze(dataset, alpha=0.05)
print(report.to_dataframe())
```

### Launch REST API

```bash
uvicorn saesV2.server.app:app --reload --port 8060
# Or via Makefile:
make start-api
```

---

## 🛠️ Repository Architecture

- `saesV2.domain`: Dataset models, test results, exception hierarchies.
- `saesV2.io`: CSV benchmark parser.
- `saesV2.validation`: Structural and value validation rules.
- `saesV2.stats`: Parametric & Welch t-tests, ANOVA, Shapiro-Wilk, Levene, non-parametric tests, Bayesian ranking, and decision engine.
- `saesV2.app`: End-to-end statistical analysis pipeline (`AnalysisReport`).
- `saesV2.reports`: Statistical tables, data exploration, power analysis, performance profiles, and FAIR reproducibility generators.
- `saesV2.visualization`: Publication-grade static and interactive plot generators.
- `saesV2.server`: FastAPI REST backend (`:8060`).
