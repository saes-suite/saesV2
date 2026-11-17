"""Integration test running full SAES evaluation on the real data.csv benchmark fixture."""

from pathlib import Path

import numpy as np

import saesV2


def test_real_dataset_fixture_end_to_end() -> None:
    fixture_path = Path(__file__).parent.parent / "fixtures" / "data.csv"
    assert fixture_path.exists(), f"Fixture file not found at {fixture_path}"

    # 1. Load dataset
    dataset = saesV2.load(fixture_path)
    assert len(dataset.df) == 14700
    assert len(dataset.algorithms) == 7
    assert len(dataset.instances) == 12
    assert len(dataset.metrics) == 7

    # 2. Validate dataset
    val_report = saesV2.validate(dataset)
    assert val_report.is_valid
    assert len(val_report.errors) == 0

    # 3. Analyze all 84 experimental cells
    analysis_report = saesV2.analyze(dataset, alpha=0.05, generate_plots=True)
    assert len(analysis_report.cell_analyses) == 84

    # 4. Verify numerical stability (no NaN statistics or p-values)
    for ca in analysis_report.cell_analyses:
        res = ca.result
        assert not np.isnan(res.statistic), f"NaN statistic in cell ({ca.instance}, {ca.metric})"
        assert not np.isnan(res.p_value), f"NaN p-value in cell ({ca.instance}, {ca.metric})"
        assert ca.figure is not None, f"Missing figure for cell ({ca.instance}, {ca.metric})"

    # 5. Export summary DataFrame
    df_summary = analysis_report.to_dataframe()
    assert len(df_summary) == 84
    assert not df_summary["statistic"].isnull().any()
    assert not df_summary["p_value"].isnull().any()
