# SAES — Scientific Statistical Evaluation Suite

Modern, high-performance Python library and API service for automated statistical analysis, hypothesis testing, and benchmark evaluation of stochastic algorithms.

---

## 📦 Installation

```bash
# 1. Core library (lightweight, zero GUI dependencies)
pip install saes

# 2. With visualization support (Matplotlib, Seaborn, Plotly)
pip install "saes[viz]"

# 3. With REST API service (FastAPI, Uvicorn)
pip install "saes[api]"

# 4. With AI Chatbot connectors (Anthropic, OpenAI)
pip install "saes[chat]"

# 5. Full installation
pip install "saes[all]"
```

---

## 🚀 Quick Usage

### Python Programmatic API

```python
import saes

# 1. Load benchmark dataset
dataset = saes.load("data.csv")

# 2. Validate format & constraints
validation = saes.validate(dataset)
assert validation.is_valid

# 3. Automated statistical analysis & test selection
report = saes.analyze(dataset, alpha=0.05)
print(report.to_dataframe())
```

### Conversational Statistical Agent

```python
from saes.chat import StatisticalChatAgent

agent = StatisticalChatAgent(report=report, dataset=dataset)
response = agent.ask("Why was Welch's ANOVA chosen for ZDT1?")
print(response)
```

### Launch REST API

```bash
uvicorn saes.server.app:app --reload --port 8060
```

---

## 🛠️ Repository Architecture

- `saes.domain`: Dataset models, test results, exception hierarchies.
- `saes.io`: CSV benchmark parser.
- `saes.validation`: Structural and value validation rules.
- `saes.stats`: Parametric & Welch t-tests, ANOVA, Shapiro-Wilk, Levene, and decision engine.
- `saes.visualization`: Publication-grade static and interactive plot generators.
- `saes.server`: FastAPI REST backend.
- `saes.chat`: Statistical AI interpretation agent.
- `saes.mcp`: Model Context Protocol server structure (prepared).
- `saes.plugins.claude`: Claude plugin and tool manifests.
