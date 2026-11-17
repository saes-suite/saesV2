"""Unit and integration tests for new scientific tabs: Profiles, Power, Publication, and Reproducibility."""

import io
import json

import pytest

import saes


@pytest.fixture
def sample_multi_instance_dataset() -> saes.Dataset:
    csv_data = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662,True
NSGAII,ZDT1,HV,1,0.661,True
NSGAII,ZDT1,HV,2,0.663,True
MOEAD,ZDT1,HV,0,0.668,True
MOEAD,ZDT1,HV,1,0.667,True
MOEAD,ZDT1,HV,2,0.669,True
SPEA2,ZDT1,HV,0,0.650,True
SPEA2,ZDT1,HV,1,0.651,True
SPEA2,ZDT1,HV,2,0.649,True
NSGAII,ZDT2,HV,0,0.440,True
NSGAII,ZDT2,HV,1,0.442,True
NSGAII,ZDT2,HV,2,0.441,True
MOEAD,ZDT2,HV,0,0.430,True
MOEAD,ZDT2,HV,1,0.432,True
MOEAD,ZDT2,HV,2,0.431,True
SPEA2,ZDT2,HV,0,0.450,True
SPEA2,ZDT2,HV,1,0.452,True
SPEA2,ZDT2,HV,2,0.451,True
"""
    return saes.load(io.StringIO(csv_data))


def test_performance_profiles_computation(sample_multi_instance_dataset: saes.Dataset) -> None:
    res = saes.compute_performance_profiles(sample_multi_instance_dataset, metric="HV", alpha=0.05)

    assert res.metric == "HV"
    assert res.maximize is True
    assert len(res.curves) == 3
    assert len(res.wtl_matrix) == 3
    assert len(res.meta_rankings) == 3

    # Check curve properties
    for curve in res.curves:
        assert len(curve.tau_values) > 0
        assert len(curve.rho_values) == len(curve.tau_values)
        assert 0.0 <= curve.efficiency <= 1.0
        assert 0.0 <= curve.robustness <= 1.0
        assert curve.robustness >= curve.efficiency

    # Check WTL matrix diagonal is neutral
    for i, row in enumerate(res.wtl_matrix):
        assert row[i].wins == 0
        assert row[i].losses == 0

    # Check meta-rankings ordering
    assert res.meta_rankings[0].rank == 1
    assert res.meta_rankings[-1].rank == 3


def test_statistical_power_and_sample_size_estimation(sample_multi_instance_dataset: saes.Dataset) -> None:
    # 1. Exact power computation
    power_high = saes.stats.power.calculate_statistical_power(n=30, effect_size=0.8, alpha=0.05)
    assert power_high > 0.85

    power_low = saes.stats.power.calculate_statistical_power(n=5, effect_size=0.2, alpha=0.05)
    assert power_low < 0.20

    # 2. A priori sample size recommendation
    rec = saes.stats.power.estimate_required_sample_size(target_power=0.80, effect_size=0.50, alpha=0.05)
    assert rec.recommended_n_independent > 20
    assert rec.recommended_n_paired < rec.recommended_n_independent
    assert rec.recommended_n_nonparametric >= rec.recommended_n_independent

    # 3. Full power analysis over dataset
    power_res = saes.power(sample_multi_instance_dataset, metric="HV", alpha=0.05)
    assert len(power_res.sample_size_recommendations) >= 3
    assert len(power_res.power_curves) >= 3
    assert len(power_res.post_hoc_audit) > 0
    assert power_res.overall_power_health in ("Optimal", "Adequate", "Underpowered")


def test_reproducibility_artifacts_generation(sample_multi_instance_dataset: saes.Dataset) -> None:
    repro = saes.reproducibility(
        sample_multi_instance_dataset,
        metric="HV",
        alpha=0.05,
        filename="benchmark.csv",
    )

    assert "import saes" in repro.python_script
    assert "saes.compute_performance_profiles" in repro.python_script
    assert "saes.analyze_statistical_power" in repro.python_script

    # Validate notebook is parseable JSON and follows nbformat 4
    nb_dict = json.loads(repro.jupyter_notebook_json)
    assert nb_dict["nbformat"] == 4
    assert len(nb_dict["cells"]) >= 6

    # Validate manifest
    assert repro.manifest.row_count == 18
    assert len(repro.manifest.dataset_checksum_sha256) == 64
