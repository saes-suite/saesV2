"""Statistical correctness verification tests against SciPy benchmarks."""

import numpy as np
import pytest
from scipy import stats

from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.decision import TestDecisionEngine
from saes.stats.ranking import BayesianRankingTest, FriedmanNemenyiTest
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def test_independent_ttest_scipy_benchmark() -> None:
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=1.0, size=30)
    g2 = np.random.normal(loc=11.5, scale=1.0, size=30)

    groups = {"AlgA": g1, "AlgB": g2}
    test = IndependentTTest()
    res = test.run(groups)

    scipy_res = stats.ttest_ind(g1, g2, equal_var=True)

    assert pytest.approx(res.statistic, abs=1e-4) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-4) == scipy_res.pvalue
    assert res.is_significant is True
    assert res.effect_size is not None
    assert res.effect_size.name == "Cohen's d"


def test_welch_ttest_scipy_benchmark() -> None:
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=1.0, size=30)
    g2 = np.random.normal(loc=11.5, scale=3.0, size=30)  # Unequal variance

    groups = {"AlgA": g1, "AlgB": g2}
    test = WelchTTest()
    res = test.run(groups)

    scipy_res = stats.ttest_ind(g1, g2, equal_var=False)

    assert pytest.approx(res.statistic, abs=1e-4) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-4) == scipy_res.pvalue
    assert res.effect_size is not None
    assert res.effect_size.name == "Hedges' g"


def test_paired_ttest_scipy_benchmark() -> None:
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=1.0, size=20)
    g2 = g1 + np.random.normal(loc=0.5, scale=0.2, size=20)  # Dependent paired data

    groups = {"AlgA": g1, "AlgB": g2}
    test = PairedTTest()
    res = test.run(groups)

    scipy_res = stats.ttest_rel(g1, g2)

    assert pytest.approx(res.statistic, abs=1e-4) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-4) == scipy_res.pvalue
    assert res.is_significant is True


def test_one_way_anova_scipy_benchmark() -> None:
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=1.0, size=25)
    g2 = np.random.normal(loc=10.2, scale=1.0, size=25)
    g3 = np.random.normal(loc=12.0, scale=1.0, size=25)

    groups = {"AlgA": g1, "AlgB": g2, "AlgC": g3}
    test = OneWayANOVA()
    res = test.run(groups)

    scipy_res = stats.f_oneway(g1, g2, g3)

    assert pytest.approx(res.statistic, abs=1e-4) == scipy_res.statistic
    assert pytest.approx(res.p_value, abs=1e-4) == scipy_res.pvalue
    assert res.effect_size is not None
    assert res.effect_size.name == "Eta-squared (η²)"


def test_welch_anova() -> None:
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=0.5, size=25)
    g2 = np.random.normal(loc=10.2, scale=2.5, size=25)
    g3 = np.random.normal(loc=12.0, scale=5.0, size=25)

    groups = {"AlgA": g1, "AlgB": g2, "AlgC": g3}
    test = WelchANOVA()
    res = test.run(groups)

    assert res.statistic > 0
    assert res.p_value < 0.05
    assert res.effect_size is not None


def test_welch_anova_zero_variance_fallback() -> None:
    g1 = np.zeros(20)
    g2 = np.zeros(20)
    np.random.seed(42)
    g3 = np.random.normal(loc=1.0, scale=0.5, size=20)

    groups = {"AlgA": g1, "AlgB": g2, "AlgC": g3}
    test = WelchANOVA()
    res = test.run(groups)

    assert not np.isnan(res.statistic)
    assert not np.isnan(res.p_value)
    assert res.statistic > 0
    assert any("zero variance" in w for w in res.warnings)


def test_assumptions_checkers() -> None:
    np.random.seed(42)
    normal_data = np.random.normal(loc=0.0, scale=1.0, size=50)
    norm_res = NormalityChecker().check({"Group1": normal_data})
    assert norm_res[0].passed is True

    g1 = np.random.normal(loc=0.0, scale=1.0, size=50)
    g2 = np.random.normal(loc=0.0, scale=1.0, size=50)
    homo_res = HomogeneityChecker().check({"Group1": g1, "Group2": g2})
    assert homo_res.passed is True


def test_decision_engine_selection() -> None:
    engine = TestDecisionEngine()
    g1 = np.random.normal(loc=0.0, scale=1.0, size=20)
    g2 = np.random.normal(loc=0.0, scale=1.0, size=20)

    # 2 independent homoscedastic groups -> IndependentTTest
    test = engine.select_test({"A": g1, "B": g2}, is_paired=False)
    assert isinstance(test, IndependentTTest)

    # 2 paired groups -> PairedTTest
    test_paired = engine.select_test({"A": g1, "B": g2}, is_paired=True)
    assert isinstance(test_paired, PairedTTest)

    # 3 homoscedastic groups -> OneWayANOVA
    g3 = np.random.normal(loc=0.0, scale=1.0, size=20)
    test_anova = engine.select_test({"A": g1, "B": g2, "C": g3})
    assert isinstance(test_anova, OneWayANOVA)


def test_friedman_nemenyi_and_bayesian_ranking() -> None:
    import pandas as pd

    rows = []
    for inst_idx in range(10):
        inst = f"ZDT{inst_idx + 1}"
        for run_id in range(5):
            rows.append(
                {
                    "algorithm": "AlgoBest",
                    "instance": inst,
                    "metric": "HV",
                    "run_id": run_id,
                    "value": 0.90,
                    "maximize": True,
                }
            )
            rows.append(
                {
                    "algorithm": "AlgoMid",
                    "instance": inst,
                    "metric": "HV",
                    "run_id": run_id,
                    "value": 0.75,
                    "maximize": True,
                }
            )
            rows.append(
                {
                    "algorithm": "AlgoWorst",
                    "instance": inst,
                    "metric": "HV",
                    "run_id": run_id,
                    "value": 0.50,
                    "maximize": True,
                }
            )

    df = pd.DataFrame(rows)
    cd_res = FriedmanNemenyiTest().run(df=df, metric="HV", alpha=0.05)
    assert cd_res.k == 3
    assert cd_res.n_blocks == 10
    assert cd_res.ranks[0].algorithm == "AlgoBest"
    assert pytest.approx(cd_res.ranks[0].avg_rank, abs=1e-5) == 1.0
    assert pytest.approx(cd_res.ranks[2].avg_rank, abs=1e-5) == 3.0
    assert cd_res.friedman_stat > 0
    assert cd_res.cd > 0

    bayes_res = BayesianRankingTest().run(
        df=df, metric="HV", rope_percent=1.0, num_samples=300, algo_a="AlgoMid", algo_b="AlgoBest"
    )
    assert bayes_res.k == 3
    assert bayes_res.reference_algo == "AlgoBest"
    assert bayes_res.items[0].algorithm == "AlgoBest"
    assert bayes_res.items[0].prob_best == 1.0
    assert bayes_res.algo_a == "AlgoMid"
    assert bayes_res.algo_b == "AlgoBest"
    assert pytest.approx(bayes_res.p_left + bayes_res.p_rope + bayes_res.p_right, abs=1e-2) == 1.0
    assert bayes_res.p_right > bayes_res.p_left
    assert len(bayes_res.points_x) == 300
    assert len(bayes_res.points_y) == 300
    assert len(bayes_res.points_density) == 300
    assert len(bayes_res.triangle_x) == 4
    assert len(bayes_res.boundary_lines) == 3
    assert bayes_res.label_rope.startswith("p(rope) = ")
    assert bayes_res.label_left.startswith("p(AlgoMid) = ")
    assert bayes_res.label_right.startswith("p(AlgoBest) = ")
    assert bayes_res.metrics_list == ["HV"]
    assert "Metric: HV" in bayes_res.metrics_label
    assert "ROPE = ±1.0%" in bayes_res.params_label
    assert bayes_res.bayesian_method == "signed-rank"

    bayes_sign_res = BayesianRankingTest().run(
        df=df,
        metric="HV",
        rope_percent=1.0,
        num_samples=300,
        algo_a="AlgoMid",
        algo_b="AlgoBest",
        method="sign",
    )
    assert bayes_sign_res.bayesian_method == "sign"
    assert "Bayesian Sign:" in bayes_sign_res.params_label
    assert bayes_sign_res.p_right > bayes_sign_res.p_left

    cd_all = FriedmanNemenyiTest().run(df=df, metric="all-metrics", alpha=0.05)
    assert cd_all.metrics_list == ["HV"]
    assert "CD =" in cd_all.params_label
