"""Integration tests for SAES high-level API and pipeline."""

import io

import saesV2


def test_full_pipeline_with_sample_data() -> None:
    csv_data = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""
    dataset = saesV2.load(io.StringIO(csv_data))
    assert len(dataset.algorithms) == 2
    assert dataset.instances == ["ZDT1"]
    assert dataset.metrics == ["HV"]

    val_report = saesV2.validate(dataset)
    assert val_report.is_valid

    analysis_report = saesV2.analyze(dataset, alpha=0.05, generate_plots=False)
    assert len(analysis_report.cell_analyses) == 1

    cell_res = analysis_report.cell_analyses[0]
    assert cell_res.instance == "ZDT1"
    assert cell_res.metric == "HV"
    assert cell_res.result.p_value < 0.05
    assert cell_res.result.is_significant is True

    df_summary = analysis_report.to_dataframe()
    assert len(df_summary) == 1
    assert "test_name" in df_summary.columns


def test_statistics_namespace_programmatic_api() -> None:
    groups = {
        "AlgA": [10.0, 10.2, 10.1, 10.3],
        "AlgB": [12.0, 12.1, 12.2, 12.0],
    }
    import numpy as np

    groups_np = {k: np.array(v) for k, v in groups.items()}

    t_res = saesV2.statistics.ttest_welch(groups_np)
    assert t_res.is_significant is True
    assert t_res.effect_size is not None

    anova_res = saesV2.statistics.anova_oneway(groups_np)
    assert anova_res.is_significant is True


def test_library_plot_and_ranking_reproducibility() -> None:
    csv_data = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT2,HV,0,0.542012,True
NSGAII,ZDT2,HV,1,0.541900,True
MOEAD,ZDT1,HV,0,0.685421,True
MOEAD,ZDT1,HV,1,0.685100,True
MOEAD,ZDT2,HV,0,0.595900,True
MOEAD,ZDT2,HV,1,0.595100,True
"""
    dataset = saesV2.load(io.StringIO(csv_data))

    cd_res = saesV2.statistics.friedman_nemenyi(dataset, metric="HV", alpha=0.05)
    assert cd_res.ranks[0].algorithm == "MOEAD"

    bayes_res = saesV2.statistics.bayesian_ranking(dataset, metric="HV", rope_percent=1.0)
    assert bayes_res.reference_algo == "MOEAD"

    fig_cd = saesV2.plot(
        dataset, plot_type="critical-distance", metric="HV", paper_format="golden", interactive=True
    )
    assert fig_cd.layout.width == 800
    assert fig_cd.layout.height == 495

    fig_bayes = saesV2.plot(
        dataset, plot_type="bayesian-rank", metric="HV", paper_format="single-col", interactive=True
    )
    assert fig_bayes.layout.width == 680
    assert fig_bayes.layout.height == 510
