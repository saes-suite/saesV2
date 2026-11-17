"""Data exploration, multi-metric reveal pivoting, normality profiling, and test recommendations."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from saes.domain.models import (
    AggregatedMetricRow,
    ContextDiagnostics,
    DataExplorationResult,
    Dataset,
    DistributionProfile,
    PivotedObservationRow,
    TestRecommendation,
)
from saes.stats.assumptions import HomogeneityChecker
from saes.stats.decision import is_paired_data


def _fmt_num(val: float) -> str:
    if not math.isfinite(val):
        return "—"
    abs_v = abs(val)
    if abs_v == 0.0:
        return "0.0000"
    if abs_v < 1e-4 or abs_v >= 1e5:
        return f"{val:.3e}"
    return f"{val:.4f}"


class DataExplorerEngine:
    """Evaluates dataset slices, reveals multi-metric tables, checks distributions, and recommends statistical tests."""

    def explore(
        self,
        dataset: Dataset,
        metrics: list[str] | None = None,
        algorithms: list[str] | None = None,
        instances: list[str] | None = None,
        alpha: float = 0.05,
    ) -> DataExplorationResult:
        df = dataset.df.copy()
        available_metrics = sorted(df["metric"].unique().tolist()) if not df.empty else []
        available_algorithms = sorted(df["algorithm"].unique().tolist()) if not df.empty else []
        available_instances = sorted(df["instance"].unique().tolist()) if not df.empty else []

        metric_directions: dict[str, bool] = {}
        for m in available_metrics:
            m_sub = df[df["metric"] == m]
            if not m_sub.empty:
                metric_directions[m] = bool(m_sub["maximize"].iloc[0])

        active_metrics = [m for m in metrics if m in available_metrics] if metrics else list(available_metrics)
        if not active_metrics and available_metrics:
            active_metrics = list(available_metrics)

        active_algorithms = (
            [a for a in algorithms if a in available_algorithms] if algorithms else list(available_algorithms)
        )
        if not active_algorithms and available_algorithms:
            active_algorithms = list(available_algorithms)

        active_instances = (
            [i for i in instances if i in available_instances] if instances else list(available_instances)
        )
        if not active_instances and available_instances:
            active_instances = list(available_instances)

        sub_df = df[
            df["metric"].isin(active_metrics)
            & df["algorithm"].isin(active_algorithms)
            & df["instance"].isin(active_instances)
        ].copy()

        # 1. Build Distribution Profiles per (instance, metric, algorithm)
        profiles: list[DistributionProfile] = []
        total_outliers = 0
        normality_pass_count = 0

        for (inst, met, alg), grp in sub_df.groupby(["instance", "metric", "algorithm"]):
            arr = grp["value"].dropna().to_numpy(dtype=float)
            n = len(arr)
            if n == 0:
                continue

            max_dir = bool(grp["maximize"].iloc[0])
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            med_v = float(np.median(arr))
            q05, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
            iqr_v = float(q75 - q25)
            min_v = float(np.min(arr))
            max_v = float(np.max(arr))
            cv_v = float(abs(std_v / mean_v)) if abs(mean_v) > 1e-12 else 0.0

            # IQR outlier detection (1.5 * IQR rule)
            if iqr_v > 0:
                lower_fence = q25 - 1.5 * iqr_v
                upper_fence = q75 + 1.5 * iqr_v
                outliers = int(np.sum((arr < lower_fence) | (arr > upper_fence)))
            else:
                outliers = 0
            total_outliers += outliers

            is_constant = bool(n >= 1 and np.all(arr == arr[0]))
            skew_v = float(stats.skew(arr)) if (n >= 3 and not is_constant) else 0.0
            kurt_v = float(stats.kurtosis(arr)) if (n >= 4 and not is_constant) else 0.0

            # Shapiro-Wilk test
            if n < 3 or is_constant:
                shapiro_stat = 1.0
                shapiro_p = 1.0
                shapiro_ok = True
            else:
                sw_res = stats.shapiro(arr)
                shapiro_stat = float(sw_res.statistic)
                shapiro_p = float(sw_res.pvalue)
                shapiro_ok = bool(shapiro_p >= alpha)

            # D'Agostino-Pearson K^2 test (requires n >= 8)
            dag_stat: float | None = None
            dag_p: float | None = None
            dag_ok: bool | None = None
            if n >= 8 and not is_constant:
                try:
                    nt_res = stats.normaltest(arr)
                    dag_stat = float(nt_res.statistic)
                    dag_p = float(nt_res.pvalue)
                    dag_ok = bool(dag_p >= alpha)
                except Exception:
                    pass

            # Kolmogorov-Smirnov test against fitted normal distribution
            if n < 3 or is_constant or std_v == 0:
                ks_stat = 0.0
                ks_p = 1.0
                ks_ok = True
            else:
                z_scores = (arr - mean_v) / std_v
                ks_res = stats.kstest(z_scores, "norm")
                ks_stat = float(ks_res.statistic)
                ks_p = float(ks_res.pvalue)
                ks_ok = bool(ks_p >= alpha)

            is_normal = shapiro_ok
            if is_normal:
                normality_pass_count += 1

            if is_constant:
                shape_label = "Constant (Zero Variance)"
            elif not is_normal:
                if skew_v > 0.8:
                    shape_label = "Right-Skewed (Non-Normal)"
                elif skew_v < -0.8:
                    shape_label = "Left-Skewed (Non-Normal)"
                elif kurt_v > 1.5:
                    shape_label = "Heavy-Tailed / Outliers (Non-Normal)"
                else:
                    shape_label = "Non-Gaussian / Multimodal"
            else:
                if abs(skew_v) > 0.5:
                    shape_label = "Approximately Normal (Mild Skew)"
                else:
                    shape_label = "Gaussian (Symmetric)"

            profiles.append(
                DistributionProfile(
                    algorithm=str(alg),
                    instance=str(inst),
                    metric=str(met),
                    maximize=max_dir,
                    n=n,
                    mean=mean_v,
                    std=std_v,
                    median=med_v,
                    iqr=iqr_v,
                    min_val=min_v,
                    max_val=max_v,
                    cv=cv_v,
                    skewness=skew_v,
                    kurtosis=kurt_v,
                    outlier_count=outliers,
                    shapiro_stat=shapiro_stat,
                    shapiro_p=shapiro_p,
                    shapiro_passed=shapiro_ok,
                    dagostino_stat=dag_stat,
                    dagostino_p=dag_p,
                    dagostino_passed=dag_ok,
                    ks_stat=ks_stat,
                    ks_p=ks_p,
                    ks_passed=ks_ok,
                    is_normal=is_normal,
                    distribution_shape=shape_label,
                    quantiles={
                        "q05": float(q05),
                        "q25": float(q25),
                        "q50": float(q50),
                        "q75": float(q75),
                        "q95": float(q95),
                    },
                    raw_values=[round(float(x), 6) for x in arr.tolist()],
                )
            )

        # 2. Build Pivoted Observation Rows (reveal all metrics side-by-side per run)
        pivoted_rows: list[PivotedObservationRow] = []
        if not sub_df.empty:
            for (alg, inst, run_id), grp in sub_df.groupby(["algorithm", "instance", "run_id"]):
                m_vals: dict[str, float] = {}
                for _, row in grp.iterrows():
                    val = float(row["value"])
                    if math.isfinite(val):
                        m_vals[str(row["metric"])] = val
                pivoted_rows.append(
                    PivotedObservationRow(
                        algorithm=str(alg),
                        instance=str(inst),
                        run_id=int(str(run_id)),
                        metrics_values=m_vals,
                    )
                )

        # 3. Build Aggregated Multi-Metric Rows per (instance, algorithm)
        aggregated_rows: list[AggregatedMetricRow] = []
        prof_lookup: dict[tuple[str, str, str], DistributionProfile] = {
            (p.instance, p.algorithm, p.metric): p for p in profiles
        }
        for inst in active_instances:
            for alg in active_algorithms:
                mean_std_map: dict[str, str] = {}
                med_iqr_map: dict[str, str] = {}
                norm_map: dict[str, bool] = {}
                shap_map: dict[str, float] = {}
                n_runs = 0
                for met in active_metrics:
                    p = prof_lookup.get((inst, alg, met))
                    if p:
                        n_runs = max(n_runs, p.n)
                        mean_std_map[met] = f"{_fmt_num(p.mean)} ± {_fmt_num(p.std)}"
                        med_iqr_map[met] = f"{_fmt_num(p.median)} ({_fmt_num(p.iqr)})"
                        norm_map[met] = p.is_normal
                        shap_map[met] = p.shapiro_p
                if n_runs > 0:
                    aggregated_rows.append(
                        AggregatedMetricRow(
                            algorithm=alg,
                            instance=inst,
                            n_runs=n_runs,
                            metrics_mean_std=mean_std_map,
                            metrics_median_iqr=med_iqr_map,
                            metrics_normal=norm_map,
                            metrics_shapiro_p=shap_map,
                        )
                    )

        # 4. Compute Context Diagnostics (Levene, Bartlett, Pairing, Balance)
        k_alg = len(active_algorithms)
        n_inst = len(active_instances)
        n_met = len(active_metrics)
        total_obs = len(sub_df)
        total_groups = len(profiles)
        norm_rate = float(normality_pass_count / total_groups) if total_groups > 0 else 1.0
        all_normal = normality_pass_count == total_groups and total_groups > 0

        counts = [p.n for p in profiles]
        runs_per_cell = int(round(float(np.median(counts)))) if counts else 0
        is_balanced = len(set(counts)) <= 1 if counts else True

        # Check pairing & homoscedasticity on primary active metric
        primary_metric = active_metrics[0] if active_metrics else ""
        primary_df = sub_df[sub_df["metric"] == primary_metric] if primary_metric else sub_df
        paired_flags: list[bool] = []
        levene_ps: list[float] = []
        levene_stats: list[float] = []
        bartlett_ps: list[float] = []
        bartlett_stats: list[float] = []

        for inst in active_instances:
            cell_df = primary_df[primary_df["instance"] == inst]
            if cell_df.empty:
                continue
            paired_flags.append(is_paired_data(cell_df))
            grp_dict: dict[str, np.ndarray] = {}
            for alg, g in cell_df.groupby("algorithm"):
                arr = g["value"].dropna().to_numpy(dtype=float)
                if len(arr) >= 2:
                    grp_dict[str(alg)] = arr
            if len(grp_dict) >= 2:
                lev = HomogeneityChecker(alpha=alpha).check(grp_dict)
                if math.isfinite(lev.p_value):
                    levene_ps.append(lev.p_value)
                    levene_stats.append(lev.statistic if math.isfinite(lev.statistic) else 0.0)
                try:
                    stds = [np.std(v, ddof=1) for v in grp_dict.values()]
                    if all(s > 0 for s in stds):
                        b_stat, b_p = stats.bartlett(*grp_dict.values())
                        if math.isfinite(b_p):
                            bartlett_ps.append(float(b_p))
                            bartlett_stats.append(float(b_stat))
                except Exception:
                    pass

        is_paired = all(paired_flags) if paired_flags else False
        min_levene_p = float(min(levene_ps)) if levene_ps else 1.0
        mean_levene_stat = float(np.mean(levene_stats)) if levene_stats else 0.0
        homo_passed = min_levene_p >= alpha
        min_bartlett_p = float(min(bartlett_ps)) if bartlett_ps else 1.0
        mean_bartlett_stat = float(np.mean(bartlett_stats)) if bartlett_stats else 0.0

        diagnostics = ContextDiagnostics(
            k_algorithms=k_alg,
            n_instances=n_inst,
            n_metrics=n_met,
            total_observations=total_obs,
            runs_per_cell=runs_per_cell,
            is_paired=is_paired,
            is_balanced=is_balanced,
            normality_pass_count=normality_pass_count,
            normality_total_groups=total_groups,
            normality_rate=norm_rate,
            all_normal=all_normal,
            homoscedasticity_levene_stat=mean_levene_stat,
            homoscedasticity_levene_p=min_levene_p,
            homoscedasticity_passed=homo_passed,
            homoscedasticity_bartlett_stat=mean_bartlett_stat,
            homoscedasticity_bartlett_p=min_bartlett_p,
            total_outliers=total_outliers,
        )

        recommendations = self._build_recommendations(diagnostics=diagnostics, alpha=alpha)

        return DataExplorationResult(
            available_metrics=available_metrics,
            available_algorithms=available_algorithms,
            available_instances=available_instances,
            metric_directions=metric_directions,
            active_metrics=active_metrics,
            active_algorithms=active_algorithms,
            active_instances=active_instances,
            alpha=alpha,
            context_diagnostics=diagnostics,
            recommendations=recommendations,
            distribution_profiles=profiles,
            pivoted_rows=pivoted_rows,
            aggregated_rows=aggregated_rows,
        )

    def _build_recommendations(
        self,
        diagnostics: ContextDiagnostics,
        alpha: float,
    ) -> list[TestRecommendation]:
        """Generate prioritized, context-aware statistical test recommendations."""
        k = diagnostics.k_algorithms
        n_inst = diagnostics.n_instances
        all_normal = diagnostics.all_normal
        norm_pct = round(diagnostics.normality_rate * 100, 1)
        homo_ok = diagnostics.homoscedasticity_passed
        is_paired = diagnostics.is_paired
        outliers = diagnostics.total_outliers

        recs: list[TestRecommendation] = []

        if k >= 3 and n_inst >= 2:
            # Multi-algorithm, multi-instance benchmark (classic Demšar 2006 / García & Herrera 2008 setting)
            recs.append(
                TestRecommendation(
                    test_id="friedman",
                    test_name="Friedman Omnibus Test + Iman-Davenport F & Holm / Nemenyi Post-Hoc",
                    category="Non-Parametric",
                    suitability="recommended",
                    suitability_score=98,
                    rationale=(
                        f"You are comparing k={k} algorithms across N={n_inst} benchmark instances "
                        f"(normality satisfied in {norm_pct}% of cells, {outliers} outlier(s) detected). "
                        "In multi-problem stochastic benchmarking, commensurability across instances cannot be "
                        "assumed and Friedman's rank test with Holm post-hoc correction is the gold standard."
                    ),
                    when_to_use=(
                        "Comparing >= 3 algorithms across multiple problem instances (blocked experimental design)."
                    ),
                    assumptions_satisfied=[
                        f"k = {k} algorithms (>= 3 required)",
                        f"N = {n_inst} matched problem instance blocks",
                        "Distribution-free (robust to non-normality and outliers)",
                    ],
                    assumptions_violated=(
                        [] if n_inst >= 5 else [f"Modest number of blocks (N={n_inst}); check Iman-Davenport F"]
                    ),
                    suggested_table_type="omnibus_posthoc",
                    posthoc_suggestion="Holm step-down (vs control) or Nemenyi Critical Distance (all-pairs)",
                    effect_size_suggestion="Kendall's W Concordance & Vargha-Delaney A12",
                    reference_citation="Demšar (2006), JMLR; García & Herrera (2008)",
                )
            )

        if k >= 2:
            wilcoxon_score = 96 if (not all_normal or n_inst >= 2 or is_paired) else 88
            recs.append(
                TestRecommendation(
                    test_id="wilcoxon",
                    test_name="Wilcoxon Signed-Rank Test (Pairwise + / ≈ / − Benchmark Table)",
                    category="Non-Parametric",
                    suitability="recommended" if wilcoxon_score >= 90 else "suitable",
                    suitability_score=wilcoxon_score,
                    rationale=(
                        f"Evaluates paired differences between algorithms across {n_inst} instance(s) "
                        f"({diagnostics.runs_per_cell} runs/cell) without assuming Gaussian residuals "
                        f"(Shapiro-Wilk normality rate: {norm_pct}%). Ideal for Instance × Algorithm (+ / ≈ / −) "
                        "and k × k pairwise significance matrices."
                    ),
                    when_to_use=(
                        "Pairwise comparison between a reference algorithm and competitors across instances or runs."
                    ),
                    assumptions_satisfied=[
                        "Matched observations across instances/runs",
                        "No normality or equal-variance assumption required",
                        "Resistant to extreme outliers",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="instance_algorithm",
                    posthoc_suggestion=f"Holm-Bonferroni p-value adjustment (α = {alpha})",
                    effect_size_suggestion="Vargha-Delaney A12 & Cliff's δ",
                    reference_citation="Demšar (2006); Derrac et al. (2011), Swarm & Evol. Comput.",
                )
            )

            mw_score = 92 if not all_normal else 82
            recs.append(
                TestRecommendation(
                    test_id="mann_whitney",
                    test_name="Mann-Whitney U Test (Wilcoxon Rank-Sum) + Vargha-Delaney A12",
                    category="Non-Parametric",
                    suitability="recommended" if (not all_normal and not is_paired) else "suitable",
                    suitability_score=mw_score,
                    rationale=(
                        f"Compares independent stochastic runs on each instance without assuming normality "
                        f"(current normality rate: {norm_pct}%). Paired with Vargha-Delaney A12, it is the standard "
                        "test for randomized algorithm runs per problem instance."
                    ),
                    when_to_use=(
                        "Comparing independent random-seed executions of algorithms on individual benchmark instances."
                    ),
                    assumptions_satisfied=[
                        f"Independent runs per cell (n ≈ {diagnostics.runs_per_cell})",
                        "Ordinal / continuous metric without Gaussian assumption",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="instance_algorithm",
                    posthoc_suggestion="Holm step-down adjustment across algorithm pairs",
                    effect_size_suggestion="Vargha-Delaney A12 (Arcuri & Briand 2014)",
                    reference_citation="Arcuri & Briand (2014), ICSE; Vargha & Delaney (2000)",
                )
            )

        if k >= 3:
            kw_score = 91 if not all_normal else 80
            recs.append(
                TestRecommendation(
                    test_id="kruskal",
                    test_name="Kruskal-Wallis H Omnibus Test + Dunn / Holm Post-Hoc",
                    category="Non-Parametric",
                    suitability="recommended" if not all_normal else "suitable",
                    suitability_score=kw_score,
                    rationale=(
                        f"Non-parametric omnibus alternative to One-Way ANOVA for k={k} algorithms per instance. "
                        f"Because {100.0 - norm_pct:.1f}% of groups depart from normality, Kruskal-Wallis avoids "
                        "Type I/II error distortion from skewed or heavy-tailed run distributions."
                    ),
                    when_to_use="Omnibus multi-group testing within individual instances when normality is violated.",
                    assumptions_satisfied=[
                        f"k = {k} independent algorithm groups per instance",
                        "Robust to non-normality and skewness",
                    ],
                    assumptions_violated=[],
                    suggested_table_type="instance_algorithm",
                    posthoc_suggestion="Pairwise Mann-Whitney / Dunn with Holm correction",
                    effect_size_suggestion="Epsilon-squared (ε²)",
                    reference_citation="Kruskal & Wallis (1952); Conover (1999)",
                )
            )

            welch_anova_score = 90 if (all_normal and not homo_ok) else (85 if all_normal else 62)
            welch_suit = "recommended" if (all_normal and not homo_ok) else ("suitable" if all_normal else "caution")
            recs.append(
                TestRecommendation(
                    test_id="anova_welch",
                    test_name="Welch's Heteroscedastic One-Way ANOVA",
                    category="Parametric",
                    suitability=welch_suit,
                    suitability_score=welch_anova_score,
                    rationale=(
                        f"Parametric omnibus test for k={k} algorithms that does NOT assume equal variances "
                        f"(Levene min p = {diagnostics.homoscedasticity_levene_p:.4f}). "
                        + (
                            "All groups satisfy Shapiro-Wilk normality."
                            if all_normal
                            else f"Caution: Normality is violated in {100.0 - norm_pct:.1f}% of groups."
                        )
                    ),
                    when_to_use="Comparing means of >= 3 normal groups when variances differ across algorithms.",
                    assumptions_satisfied=(
                        ["Robust to unequal variances (heteroscedasticity)"]
                        + (["Shapiro-Wilk normality satisfied"] if all_normal else [])
                    ),
                    assumptions_violated=(
                        [] if all_normal else [f"Normality violated in {100.0 - norm_pct:.1f}% of groups"]
                    ),
                    suggested_table_type="instance_algorithm",
                    posthoc_suggestion="Games-Howell / Welch t-test with Holm adjustment",
                    effect_size_suggestion="Eta-squared (η²)",
                    reference_citation="Welch (1951), Biometrika",
                )
            )

            anova_score = 93 if (all_normal and homo_ok) else (50 if not all_normal else 58)
            levene_viol = f"Homoscedasticity violated (Levene p={diagnostics.homoscedasticity_levene_p:.4f})"
            recs.append(
                TestRecommendation(
                    test_id="anova_oneway",
                    test_name="Fisher's One-Way ANOVA",
                    category="Parametric",
                    suitability="recommended" if (all_normal and homo_ok) else "caution",
                    suitability_score=anova_score,
                    rationale=(
                        f"Classic parametric omnibus F-test across k={k} algorithms. "
                        + (
                            "Both Gaussian normality and Levene's homoscedasticity are satisfied."
                            if (all_normal and homo_ok)
                            else (
                                "Requires both normality and equal variances; "
                                "consider Kruskal-Wallis or Welch's ANOVA when violated."
                            )
                        )
                    ),
                    when_to_use="Comparing means of >= 3 algorithms when residuals are normal and homoscedastic.",
                    assumptions_satisfied=(
                        (["Shapiro-Wilk normality satisfied"] if all_normal else [])
                        + (["Levene's equal variance satisfied"] if homo_ok else [])
                    ),
                    assumptions_violated=(
                        ([] if all_normal else [f"Normality violated ({norm_pct}% pass rate)"])
                        + ([] if homo_ok else [levene_viol])
                    ),
                    suggested_table_type="instance_algorithm",
                    posthoc_suggestion="Tukey's HSD or Holm-adjusted t-tests",
                    effect_size_suggestion="Eta-squared (η²) & Omega-squared (ω²)",
                    reference_citation="Fisher (1925)",
                )
            )

        if k == 2:
            welch_t_score = 94 if (all_normal and not homo_ok) else (88 if all_normal else 65)
            welch_t_suit = (
                "recommended" if (all_normal and not is_paired) else ("suitable" if all_normal else "caution")
            )
            recs.append(
                TestRecommendation(
                    test_id="ttest_welch",
                    test_name="Welch's Two-Sample t-test (Unequal Variances)",
                    category="Parametric",
                    suitability=welch_t_suit,
                    suitability_score=welch_t_score,
                    rationale=(
                        "Compares means of 2 algorithms using Satterthwaite degrees of freedom "
                        "without requiring equal variances."
                    ),
                    when_to_use="Two-algorithm parametric comparison with potentially unequal variances.",
                    assumptions_satisfied=(
                        ["Robust to unequal variances"] + (["Normality satisfied"] if all_normal else [])
                    ),
                    assumptions_violated=[] if all_normal else [f"Normality violated ({norm_pct}% pass rate)"],
                    suggested_table_type="instance_algorithm",
                    posthoc_suggestion="Not required (k = 2)",
                    effect_size_suggestion="Hedges' g",
                    reference_citation="Welch (1947)",
                )
            )

        # Bayesian ROPE Simplex Recommendation (always suitable & modern state-of-the-art)
        recs.append(
            TestRecommendation(
                test_id="bayesian_rope",
                test_name="Bayesian Signed-Rank Simplex ROPE & Posterior Ranking",
                category="Bayesian",
                suitability="recommended" if k >= 2 else "suitable",
                suitability_score=91,
                rationale=(
                    "State-of-the-art Bayesian non-parametric alternative to NHST p-values. Computes posterior "
                    "probabilities P(A > B), P(ROPE practical equivalence), and P(B > A) via Dirichlet Process."
                ),
                when_to_use=(
                    "When you want direct posterior probabilities of practical superiority vs equivalence (ROPE) "
                    "instead of null-hypothesis p-values."
                ),
                assumptions_satisfied=[
                    "Non-parametric Dirichlet Process prior",
                    "Immune to sample-size p-value inflation",
                    "Quantifies practical equivalence (ROPE)",
                ],
                assumptions_violated=[],
                suggested_table_type="pairwise_matrix",
                posthoc_suggestion="Posterior probability simplex (no family-wise alpha penalty)",
                effect_size_suggestion="ROPE Probability P(|Δ| ≤ ROPE)",
                reference_citation="Benavoli et al. (2017), JMLR",
            )
        )

        recs.sort(key=lambda r: r.suitability_score, reverse=True)
        return recs


def _fmt_opt_num(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return _fmt_num(val)


def _fmt_p(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    if val < 0.0001:
        return f"{val:.2e}"
    return f"{val:.4f}"


def export_exploration_table(
    exploration: DataExplorationResult,
    view_mode: str = "reveal_metrics",
    export_format: str = "csv",
    stat_format: str = "mean_std",
    normality_filter: str = "all",
) -> tuple[bytes, str, str]:
    """Export an Observations & Data exploration view into PDF, DOCX, LaTeX, CSV, Markdown, or HTML.

    Args:
        exploration: DataExplorationResult produced by DataExplorerEngine.explore().
        view_mode: One of 'reveal_metrics' (aggregated), 'by_run' (pivoted runs),
                   'normality' (distribution diagnostics), or 'recommendations'.
        export_format: One of 'pdf', 'docx', 'latex', 'csv', 'markdown', 'html'.
        stat_format: 'mean_std' or 'median_iqr' for the 'reveal_metrics' view.
        normality_filter: 'all', 'normal', 'non_normal', or 'outliers' for the 'normality' view.

    Returns:
        Tuple of (file_bytes, media_type, extension).
    """
    from saes.stats.tables import export_flat_table

    mode = view_mode.strip().lower()
    metrics = exploration.active_metrics

    if mode == "by_run":
        headers = ["Algorithm", "Instance", "Run ID"] + metrics
        rows: list[list[str]] = []
        bolds: list[list[bool]] = []
        for r in exploration.pivoted_rows:
            row_vals = [r.algorithm, r.instance, str(r.run_id)] + [
                _fmt_opt_num(r.metrics_values.get(m)) for m in metrics
            ]
            rows.append(row_vals)
            bolds.append([False] * len(headers))
        title = "SAES Observations & Multi-Metric Run Explorer"
        caption = (
            f"Pivoted per-run observations across {len(exploration.active_algorithms)} algorithm(s), "
            f"{len(exploration.active_instances)} instance(s), and {len(metrics)} revealed metric(s)."
        )
        return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)

    if mode == "normality":
        headers = [
            "Metric",
            "Instance",
            "Algorithm",
            "n",
            "Mean ± SD",
            "Median (IQR)",
            "Skewness",
            "Kurtosis",
            "Shapiro p",
            "KS p",
            "Outliers",
            "Shape",
            "Suggested Paradigm",
        ]
        rows = []
        bolds = []
        for p in exploration.distribution_profiles:
            if normality_filter == "normal" and not p.is_normal:
                continue
            if normality_filter == "non_normal" and p.is_normal:
                continue
            if normality_filter == "outliers" and p.outlier_count == 0:
                continue
            paradigm = "Parametric / Non-Parametric" if p.is_normal else "Non-Parametric (Rank-Based)"
            rows.append(
                [
                    p.metric,
                    p.instance,
                    p.algorithm,
                    str(p.n),
                    f"{_fmt_num(p.mean)} ± {_fmt_num(p.std)}",
                    f"{_fmt_num(p.median)} ({_fmt_num(p.iqr)})",
                    f"{p.skewness:+.2f}",
                    f"{p.kurtosis:+.2f}",
                    _fmt_p(p.shapiro_p),
                    _fmt_p(p.ks_p),
                    str(p.outlier_count),
                    p.distribution_shape,
                    paradigm,
                ]
            )
            bolds.append([p.is_normal] * len(headers))
        title = "SAES Normality & Distribution Diagnostics Table"
        caption = (
            f"Shapiro-Wilk, D'Agostino-Pearson K², Kolmogorov-Smirnov, skewness, kurtosis, and IQR outlier "
            f"diagnostics across {len(rows)} cell distribution(s)."
        )
        return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)

    if mode == "recommendations":
        headers = [
            "Suitability",
            "Score",
            "Statistical Test",
            "Category",
            "Post-Hoc Correction",
            "Effect Size",
            "Reference",
        ]
        rows = []
        bolds = []
        for rec in exploration.recommendations:
            is_rec = rec.suitability == "recommended"
            rows.append(
                [
                    rec.suitability.upper(),
                    f"{rec.suitability_score}/100",
                    rec.test_name,
                    rec.category,
                    rec.posthoc_suggestion,
                    rec.effect_size_suggestion,
                    rec.reference_citation,
                ]
            )
            bolds.append([is_rec] * len(headers))
        title = "SAES Context-Aware Statistical Test Recommendations"
        diag = exploration.context_diagnostics
        caption = (
            f"Context-aware recommendations for {diag.k_algorithms} algorithm(s), {diag.n_instances} instance(s), "
            f"and {diag.n_metrics} metric(s) (normality pass rate: {diag.normality_rate * 100:.1f}%)."
        )
        return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)

    # Default: "reveal_metrics" (Multi-Metric Aggregated Table)
    stat_label = "Median (IQR)" if stat_format == "median_iqr" else "Mean ± SD"
    headers = ["Algorithm", "Instance", "Runs (n)"]
    for m in metrics:
        headers.append(f"{m} ({stat_label})")
        headers.append(f"{m} Normality")

    rows = []
    bolds = []
    for agg in exploration.aggregated_rows:
        r_vals = [agg.algorithm, agg.instance, str(agg.n_runs)]
        b_vals = [True, False, False]
        for m in metrics:
            if stat_format == "median_iqr":
                val_str = agg.metrics_median_iqr.get(m, "—")
            else:
                val_str = agg.metrics_mean_std.get(m, "—")
            norm_ok = agg.metrics_normal.get(m)
            norm_str = "Normal" if norm_ok is True else ("Non-Normal" if norm_ok is False else "—")
            r_vals.extend([val_str, norm_str])
            b_vals.extend([False, bool(norm_ok)])
        rows.append(r_vals)
        bolds.append(b_vals)

    title = f"SAES Multi-Metric Reveal Table ({stat_label})"
    caption = (
        f"Side-by-side multi-metric comparison across {len(exploration.active_algorithms)} algorithm(s), "
        f"{len(exploration.active_instances)} instance(s), and {len(metrics)} revealed metric(s)."
    )
    return export_flat_table(headers, rows, bolds, title, caption, export_format=export_format)
