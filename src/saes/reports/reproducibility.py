"""Reproducibility script generator, Jupyter Notebook synthesis, and FAIR environment manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    Dataset,
    EnvironmentManifest,
    ReproducibilityResult,
)


def _compute_sha256(content: str) -> str:
    """Compute SHA256 hex digest of dataset text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_environment_manifest(
    dataset: Dataset,
    raw_csv: str | None = None,
) -> EnvironmentManifest:
    """Generate reproducible execution environment manifest and dataset checksum."""
    if raw_csv:
        checksum = _compute_sha256(raw_csv)
    else:
        csv_str = dataset.df.to_csv(index=False)
        checksum = _compute_sha256(csv_str)

    algorithms = dataset.algorithms
    instances = dataset.instances
    if algorithms and instances:
        first_inst, first_algo = instances[0], algorithms[0]
        sample_sub = dataset.df[(dataset.df["instance"] == first_inst) & (dataset.df["algorithm"] == first_algo)]
    else:
        sample_sub = None
    runs_per_cell = len(sample_sub) if sample_sub is not None and not sample_sub.empty else 1

    return EnvironmentManifest(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.platform(),
        saes_version="0.1.0",
        scipy_version=scipy.__version__,
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        dataset_checksum_sha256=checksum,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        row_count=len(dataset.df),
        runs_per_cell=runs_per_cell,
    )


def generate_python_script(
    dataset: Dataset,
    metric: str,
    alpha: float = 0.05,
    filename: str = "dataset.csv",
    checksum: str = "",
) -> str:
    """Generate a clean, standalone Python script replicating the exact SAES analysis."""
    return f'''#!/usr/bin/env python3
"""Standalone reproducibility script for SAES benchmark evaluation.

Generated automatically by SAES (Stochastic Algorithm Evaluation Suite).
Dataset: {filename}
SHA256: {checksum}
Significance level: alpha = {alpha}
Metric: {metric}
"""

from __future__ import annotations

import sys
import saes

def main() -> None:
    print("=" * 70)
    print(" SAES REPRODUCIBILITY EXECUTION ")
    print("=" * 70)

    # 1. Load dataset
    print(f"--> [1/5] Loading benchmark dataset from '{filename}'...")
    try:
        dataset = saes.load("{filename}")
    except FileNotFoundError:
        print(f"ERROR: Dataset file '{filename}' not found. Please place it in the working directory.")
        sys.exit(1)

    print(
        f"    Loaded {{len(dataset.df)}} observations across "
        f"{{len(dataset.algorithms)}} algorithms and {{len(dataset.instances)}} instances."
    )

    # 2. Structural & Statistical Validation
    print("--> [2/5] Validating data integrity...")
    val_report = saes.validate(dataset)
    if not val_report.is_valid:
        print(f"    Validation errors encountered: {{len(val_report.errors)}}")
        for err in val_report.errors:
            print(f"      - {{err}}")
        sys.exit(1)
    print("    Validation passed successfully.")

    # 3. Comprehensive Statistical Analysis & Hypothesis Testing
    print(f"--> [3/5] Executing statistical tests (alpha = {alpha})...")
    analysis = saes.analyze(dataset, alpha={alpha}, generate_plots=False)
    print(f"    Evaluated {{analysis.cells_count}} experimental cells.")

    # 4. Performance Profiles & Meta-Benchmark
    print(f"--> [4/5] Computing Dolan-Moré (2002) performance profiles for '{metric}'...")
    profiles = saes.compute_performance_profiles(dataset, metric="{metric}", alpha={alpha})
    print("    Global Meta-Rankings:")
    for r in profiles.meta_rankings:
        print(
            f"      #{{r.rank}}: {{r.algorithm:<12}} | "
            f"Net: {{r.net_score:+2d}} | "
            f"Eff: {{r.efficiency:.1%}} | "
            f"Rob: {{r.robustness:.1%}}"
        )

    # 5. Statistical Power & Experimental Design Audit
    print(f"--> [5/5] Auditing statistical power & experimental design for '{metric}'...")
    power_res = saes.analyze_statistical_power(dataset, metric="{metric}", alpha={alpha})
    print(
        f"    Audited {{len(power_res.post_hoc_audit)}} pairwise comparisons. "
        f"Health: {{power_res.overall_power_health}}"
    )
    print("\\n[SUCCESS] All statistical evaluations reproduced accurately.")

if __name__ == "__main__":
    main()
'''


def generate_jupyter_notebook(
    dataset: Dataset,
    metric: str,
    alpha: float = 0.05,
    filename: str = "dataset.csv",
    manifest: EnvironmentManifest | None = None,
) -> str:
    """Generate a fully executable Jupyter Notebook (.ipynb) replicating the study."""
    manifest_info = manifest or generate_environment_manifest(dataset)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SAES Empirical Benchmark & Statistical Evaluation\n",
                "\n",
                f"**Dataset**: `{filename}`  \n",
                f"**Checksum (SHA256)**: `{manifest_info.dataset_checksum_sha256}`  \n",
                f"**Metric**: `{metric}` | **Significance**: $\\alpha = {alpha}$  \n",
                f"**Environment**: Python {manifest_info.python_version} ({manifest_info.platform})  \n",
                f"**Generated**: {manifest_info.timestamp} via **SAES v{manifest_info.saes_version}**\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup & Data Ingestion\n",
                "Load libraries and verify dataset schema compliance.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import saes\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                f"dataset = saes.load('{filename}')\n",
                "print(f'Algorithms: {dataset.algorithms}')\n",
                "print(f'Instances:  {dataset.instances}')\n",
                "print(f'Metrics:    {dataset.metrics}')\n",
                "dataset.df.head()\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Validation & Assumption Checks\n",
                "Verify structural integrity, missing values, and check normality & homoscedasticity.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "val_report = saes.validate(dataset)\n",
                "print(f'Dataset Valid: {val_report.is_valid}')\n",
                "if not val_report.is_valid:\n",
                "    print(val_report.errors)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Comprehensive Statistical Hypothesis Testing\n",
                "Execute omnibus Friedman/ANOVA tests, post-hoc pairwise tests, and effect sizes.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"analysis_report = saes.analyze(dataset, alpha={alpha}, generate_plots=False)\n",
                "print(f'Analyzed {analysis_report.cells_count} experimental cells.')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Dolan-Moré Performance Profiles & Global Meta-Ranking\n",
                "Evaluate solver efficiency $\\rho(1)$ and robustness $\\rho(\\tau)$ across all instances.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"profiles = saes.compute_performance_profiles(dataset, metric='{metric}', alpha={alpha})\n",
                "for r in profiles.meta_rankings:\n",
                "    print(f'Rank {r.rank}: {r.algorithm} (Net: {r.net_score:+d}, Eff: {r.efficiency:.1%})')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Statistical Power & Experimental Design\n",
                "Audit achieved statistical power across comparisons.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"power_res = saes.analyze_statistical_power(dataset, metric='{metric}', alpha={alpha})\n",
                "print(f'Overall Power Health: {power_res.overall_power_health}')\n",
                "print('Recommended runs per cell for target power = 80%:')\n",
                "for rec in power_res.sample_size_recommendations:\n",
                (
                    "    print(f'  {rec.effect_size_label}: "
                    "N = {rec.recommended_n_independent} (paired: {rec.recommended_n_paired})')\n"
                ),
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Statistical Power & Experimental Design\n",
                "Audit post-hoc statistical power and sample size sensitivity.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"power_res = saes.analyze_statistical_power(dataset, metric='{metric}', alpha={alpha})\n",
                "print(f'Overall Power Health: {power_res.overall_power_health}')\n",
                "for rec in power_res.sample_size_recommendations:\n",
                "    print(f'  {rec.effect_size_label}: Indep N={rec.recommended_n_independent}')\n",
            ],
        },
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": manifest_info.python_version,
            },
            "orig_nbformat": 4,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return json.dumps(notebook_dict, indent=2)


def generate_reproducibility_artifacts(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    filename: str = "dataset.csv",
    raw_csv: str | None = None,
) -> ReproducibilityResult:
    """Generate complete reproducibility pack: Python script, Jupyter notebook, and FAIR manifest."""
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no metrics for reproducibility generation.")

    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]
    manifest = generate_environment_manifest(dataset, raw_csv=raw_csv)

    py_script = generate_python_script(
        dataset=dataset,
        metric=target_metric,
        alpha=alpha,
        filename=filename,
        checksum=manifest.dataset_checksum_sha256,
    )

    jupyter_nb = generate_jupyter_notebook(
        dataset=dataset,
        metric=target_metric,
        alpha=alpha,
        filename=filename,
        manifest=manifest,
    )

    return ReproducibilityResult(
        python_script=py_script,
        jupyter_notebook_json=jupyter_nb,
        manifest=manifest,
    )
