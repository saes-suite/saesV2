.PHONY: help install start-api build lint format test ci version

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install saes with all extras
	uv sync --extra dev

start-api: ## Launch FastAPI REST server
	uv run uvicorn saes.server.app:app --reload --port 8000

build: ## Build wheel and sdist packages
	uv build

lint: ## Run ruff and mypy
	uv run ruff check .
	uv run mypy src

format: ## Auto-format code
	uv run ruff format .

test: ## Run test suite
	uv run pytest

ci: lint test ## Run full CI quality gate
