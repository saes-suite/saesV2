"""Scientific benchmark table generation (ANOVA, Friedman, Wilcoxon, Kruskal-Wallis) and multi-format export."""

from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from saesV2.domain.exceptions import StatisticalAnalysisError
from saesV2.domain.models import (
    AlgorithmSummaryFooter,
    BenchmarkTableCell,
    BenchmarkTableInstanceRow,
    Dataset,
    PairwiseInstanceComparison,
    PairwiseMatrixCell,
    StatisticalResult,
    StatisticalTableResult,
)
from saesV2.stats.anova import OneWayANOVA, WelchANOVA
from saesV2.stats.assumptions import HomogeneityChecker, NormalityChecker
from saesV2.stats.decision import is_paired_data
from saesV2.stats.nonparametric import (
    KruskalWallisTest,
    MannWhitneyUTest,
    WilcoxonSignedRankTest,
    adjust_pvalues,
    compute_cliffs_delta,
    compute_vargha_delaney_a12,
)
from saesV2.stats.ranking import FriedmanNemenyiTest
from saesV2.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def _fast_bayesian_pair_probs(
    vec_r: np.ndarray,
    vec_c: np.ndarray,
    maximize: bool,
    rope_percent: float = 1.0,
) -> tuple[float, float, float]:
    """Compute fast Bayesian posterior ROPE probabilities (P(row better), P(ROPE), P(col better)) without KDE."""
    r_clean = vec_r[~np.isnan(vec_r)]
    c_clean = vec_c[~np.isnan(vec_c)]
    if len(r_clean) == 0 or len(c_clean) == 0:
        return 0.0, 1.0, 0.0

    combined = np.concatenate([r_clean, c_clean])
    val_range = float(np.max(combined) - np.min(combined))
    rope_half = max(1e-9, val_range * (rope_percent / 100.0))

    if len(r_clean) == len(c_clean) and len(r_clean) >= 2:
        diffs = (r_clean - c_clean) if maximize else (c_clean - r_clean)
        n = len(diffs)
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        if sd_d <= 1e-12:
            if abs(mean_d) <= rope_half:
                return 0.0, 1.0, 0.0
            return (1.0, 0.0, 0.0) if mean_d > 0 else (0.0, 0.0, 1.0)
        se_d = sd_d / math.sqrt(n)
        cdf_low = float(stats.t.cdf(-rope_half, df=n - 1, loc=mean_d, scale=se_d))
        cdf_high = float(stats.t.cdf(rope_half, df=n - 1, loc=mean_d, scale=se_d))
        p_col = max(0.0, min(1.0, cdf_low))
        p_rope = max(0.0, min(1.0, cdf_high - cdf_low))
        p_row = max(0.0, min(1.0, 1.0 - cdf_high))
        return p_row, p_rope, p_col

    mean_r, mean_c = float(np.mean(r_clean)), float(np.mean(c_clean))
    diff_m = (mean_r - mean_c) if maximize else (mean_c - mean_r)
    v_r = float(np.var(r_clean, ddof=1)) / len(r_clean) if len(r_clean) > 1 else 0.0
    v_c = float(np.var(c_clean, ddof=1)) / len(c_clean) if len(c_clean) > 1 else 0.0
    se_pool = math.sqrt(v_r + v_c)
    if se_pool <= 1e-12:
        if abs(diff_m) <= rope_half:
            return 0.0, 1.0, 0.0
        return (1.0, 0.0, 0.0) if diff_m > 0 else (0.0, 0.0, 1.0)
    cdf_low = float(stats.norm.cdf(-rope_half, loc=diff_m, scale=se_pool))
    cdf_high = float(stats.norm.cdf(rope_half, loc=diff_m, scale=se_pool))
    p_col = max(0.0, min(1.0, cdf_low))
    p_rope = max(0.0, min(1.0, cdf_high - cdf_low))
    p_row = max(0.0, min(1.0, 1.0 - cdf_high))
    return p_row, p_rope, p_col


def _fast_bayesian_summary_probs(
    inst_means_matrix: dict[str, np.ndarray],
    algorithms: list[str],
    ref_algo: str,
    maximize: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute fast Bayesian P(Best), P(Better than Ref), and P(ROPE with Ref) across algorithms."""
    bayes_best_map: dict[str, float] = {}
    bayes_better_map: dict[str, float] = {}
    bayes_rope_map: dict[str, float] = {}

    ref_vec = inst_means_matrix.get(ref_algo, np.array([], dtype=float))
    for alg in algorithms:
        if alg == ref_algo:
            bayes_better_map[alg] = 0.0
            bayes_rope_map[alg] = 1.0
        else:
            alg_vec = inst_means_matrix.get(alg, np.array([], dtype=float))
            p_b, p_r, _ = _fast_bayesian_pair_probs(alg_vec, ref_vec, maximize=maximize)
            bayes_better_map[alg] = p_b
            bayes_rope_map[alg] = p_r

    rng = np.random.default_rng(42)
    k = len(algorithms)
    means = np.array(
        [float(np.mean(inst_means_matrix[a])) if len(inst_means_matrix.get(a, [])) > 0 else 0.0 for a in algorithms],
        dtype=float,
    )
    ses = np.array(
        [
            (
                float(np.std(inst_means_matrix[a], ddof=1)) / math.sqrt(len(inst_means_matrix[a]))
                if len(inst_means_matrix.get(a, [])) > 1
                else 1e-6
            )
            for a in algorithms
        ],
        dtype=float,
    )
    draws = rng.normal(loc=means[:, None], scale=np.maximum(ses[:, None], 1e-9), size=(k, 500))
    winners = np.argmax(draws, axis=0) if maximize else np.argmin(draws, axis=0)
    counts = np.bincount(winners, minlength=k) / 500.0
    for idx, alg in enumerate(algorithms):
        bayes_best_map[alg] = float(counts[idx])

    return bayes_best_map, bayes_better_map, bayes_rope_map


def _format_num(val: float, precision: int = 4) -> str:
    """Format a numeric value cleanly for scientific tables."""
    if not math.isfinite(val):
        return "—"
    abs_v = abs(val)
    if abs_v == 0.0:
        return f"{0.0:.{precision}f}"
    if abs_v < 1e-4 or abs_v >= 1e5:
        return f"{val:.{max(2, precision - 1)}e}"
    return f"{val:.{precision}f}"


def _format_p(p_val: float | None) -> str:
    """Format a p-value for scientific reporting."""
    if p_val is None or not math.isfinite(p_val):
        return "—"
    if p_val < 0.0001:
        return f"{p_val:.2e}"
    return f"{p_val:.4f}"


def _compute_cohens_d_raw(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute signed Cohen's d between two 1D arrays."""
    c1 = x1[~np.isnan(x1)]
    c2 = x2[~np.isnan(x2)]
    n1, n2 = len(c1), len(c2)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1 = float(np.var(c1, ddof=1))
    v2 = float(np.var(c2, ddof=1))
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(c1) - np.mean(c2)) / math.sqrt(pooled_var))


def _run_omnibus_for_groups(
    groups: dict[str, np.ndarray],
    test_type: str,
    alpha: float,
    is_paired: bool = False,
) -> StatisticalResult:
    """Execute the requested or auto-selected omnibus/group test for a single instance."""
    k = len(groups)
    norm_type = test_type.strip().lower()

    if k == 2:
        if norm_type in {"wilcoxon", "wilcoxon_signed_rank", "friedman"} and is_paired:
            return WilcoxonSignedRankTest().run(groups, alpha=alpha)
        if norm_type in {"mann_whitney", "kruskal", "wilcoxon", "friedman"}:
            return MannWhitneyUTest().run(groups, alpha=alpha)
        if norm_type in {"ttest_rel", "paired_ttest"} and is_paired:
            return PairedTTest().run(groups, alpha=alpha)
        if norm_type in {"ttest_ind", "anova_oneway", "anova"}:
            return IndependentTTest().run(groups, alpha=alpha)
        if norm_type in {"ttest_welch", "anova_welch", "welch"}:
            return WelchTTest().run(groups, alpha=alpha)

        # Auto mode for k=2
        norm_checks = NormalityChecker(alpha=alpha).check(groups)
        all_normal = all(n.passed for n in norm_checks)
        if not all_normal:
            if is_paired:
                return WilcoxonSignedRankTest().run(groups, alpha=alpha)
            return MannWhitneyUTest().run(groups, alpha=alpha)
        if is_paired:
            return PairedTTest().run(groups, alpha=alpha)
        homo = HomogeneityChecker(alpha=alpha).check(groups)
        return IndependentTTest().run(groups, alpha=alpha) if homo.passed else WelchTTest().run(groups, alpha=alpha)

    # k > 2
    if norm_type in {"anova_oneway", "anova", "ttest_ind"}:
        return OneWayANOVA().run(groups, alpha=alpha)
    if norm_type in {"anova_welch", "welch", "ttest_welch"}:
        return WelchANOVA().run(groups, alpha=alpha)
    if norm_type in {"kruskal", "kruskal_wallis", "wilcoxon", "mann_whitney", "friedman"}:
        return KruskalWallisTest().run(groups, alpha=alpha)

    # Auto mode for k > 2: check normality + homoscedasticity
    norm_checks = NormalityChecker(alpha=alpha).check(groups)
    all_normal = all(n.passed for n in norm_checks)
    if not all_normal:
        return KruskalWallisTest().run(groups, alpha=alpha)
    homo = HomogeneityChecker(alpha=alpha).check(groups)
    return OneWayANOVA().run(groups, alpha=alpha) if homo.passed else WelchANOVA().run(groups, alpha=alpha)


def _run_pairwise_test(
    x_algo: np.ndarray,
    x_ref: np.ndarray,
    algo_name: str,
    ref_name: str,
    test_type: str,
    alpha: float,
    is_paired: bool,
) -> StatisticalResult:
    """Run a two-sample test between an algorithm and a reference/comparison algorithm."""
    groups = {algo_name: x_algo, ref_name: x_ref}
    norm_type = test_type.strip().lower()
    equal_len = len(x_algo) == len(x_ref) and is_paired

    if norm_type in {"wilcoxon", "wilcoxon_signed_rank", "friedman"}:
        if equal_len:
            return WilcoxonSignedRankTest().run(groups, alpha=alpha)
        return MannWhitneyUTest().run(groups, alpha=alpha)

    if norm_type in {"mann_whitney", "kruskal", "kruskal_wallis"}:
        return MannWhitneyUTest().run(groups, alpha=alpha)

    if norm_type in {"ttest_rel", "paired_ttest"}:
        if equal_len:
            return PairedTTest().run(groups, alpha=alpha)
        return WelchTTest().run(groups, alpha=alpha)

    if norm_type in {"ttest_ind", "anova_oneway", "anova"}:
        return IndependentTTest().run(groups, alpha=alpha)

    if norm_type in {"ttest_welch", "anova_welch", "welch"}:
        return WelchTTest().run(groups, alpha=alpha)

    # Auto mode: check normality of both groups
    norm_checks = NormalityChecker(alpha=alpha).check(groups)
    both_normal = all(n.passed for n in norm_checks)
    if not both_normal:
        if equal_len:
            return WilcoxonSignedRankTest().run(groups, alpha=alpha)
        return MannWhitneyUTest().run(groups, alpha=alpha)

    if equal_len:
        return PairedTTest().run(groups, alpha=alpha)
    homo = HomogeneityChecker(alpha=alpha).check(groups)
    return IndependentTTest().run(groups, alpha=alpha) if homo.passed else WelchTTest().run(groups, alpha=alpha)


class StatisticalTableGenerator:
    """Generates publication-ready ANOVA, Friedman, Wilcoxon, and Pairwise statistical benchmark tables."""

    def generate(
        self,
        dataset: Dataset,
        metric: str | None = None,
        table_type: str = "instance_algorithm",
        test_type: str = "auto",
        posthoc_correction: str = "holm",
        summary_stat: str = "auto",
        reference_algorithm: str | None = None,
        scope: str = "all-instances",
        selected_instance: str | None = None,
        algorithms_filter: list[str] | None = None,
        instances_filter: list[str] | None = None,
        latex_document_mode: str = "standalone",
        latex_notation: str = "scientific",
        alpha: float = 0.05,
        precision: int = 4,
    ) -> StatisticalTableResult:
        """Generate a complete StatisticalTableResult with cells, omnibus tests, pairwise matrices, and previews."""
        df = dataset.df.copy()
        if df.empty:
            raise StatisticalAnalysisError("Cannot generate statistical table from an empty dataset.")

        available_metrics = sorted(df["metric"].unique().tolist())
        target_metric = metric if (metric and metric in available_metrics) else available_metrics[0]
        sub_df = df[df["metric"] == target_metric].copy()

        if algorithms_filter:
            valid_algos = [a for a in algorithms_filter if a in sub_df["algorithm"].unique()]
            if len(valid_algos) >= 2:
                sub_df = sub_df[sub_df["algorithm"].isin(valid_algos)].copy()

        if instances_filter:
            valid_insts = [i for i in instances_filter if i in sub_df["instance"].unique()]
            if valid_insts:
                sub_df = sub_df[sub_df["instance"].isin(valid_insts)].copy()

        if scope == "single-instance" and selected_instance:
            if selected_instance in sub_df["instance"].unique():
                sub_df = sub_df[sub_df["instance"] == selected_instance].copy()

        algorithms = sorted(sub_df["algorithm"].unique().tolist())
        instances = sorted(sub_df["instance"].unique().tolist())
        if len(algorithms) < 2:
            raise StatisticalAnalysisError(
                f"Statistical table generation requires at least 2 algorithms, got {len(algorithms)}."
            )

        maximize = bool(sub_df["maximize"].iloc[0])

        # 1. Global Friedman + Nemenyi and Bayesian Ranking across the selected subset
        friedman_stat: float | None = None
        friedman_p: float | None = None
        iman_f: float | None = None
        iman_p: float | None = None
        kendalls_w: float | None = None
        nemenyi_cd: float | None = None
        global_ranks_map: dict[str, float] = {}
        bayes_best_map: dict[str, float] = {}
        bayes_better_map: dict[str, float] = {}
        bayes_rope_map: dict[str, float] = {}

        effective_scope = "single-instance" if len(instances) == 1 else "all-instances"
        effective_inst = instances[0] if len(instances) == 1 else None

        try:
            cd_res = FriedmanNemenyiTest().run(
                df=sub_df,
                metric=target_metric,
                alpha=alpha,
                scope=effective_scope,
                instance=effective_inst,
            )
            friedman_stat = cd_res.friedman_stat
            friedman_p = cd_res.friedman_p_value
            iman_f = cd_res.iman_davenport_f
            iman_p = cd_res.iman_davenport_p_value
            nemenyi_cd = cd_res.cd
            if cd_res.n_blocks > 0 and cd_res.k > 1:
                kendalls_w = float(min(1.0, max(0.0, cd_res.friedman_stat / (cd_res.n_blocks * (cd_res.k - 1)))))
            for r_item in cd_res.ranks:
                global_ranks_map[r_item.algorithm] = r_item.avg_rank
        except Exception:
            for idx, alg in enumerate(algorithms):
                global_ranks_map[alg] = float(idx + 1)

        # Determine Reference / Control Algorithm (defaults to #1 best Friedman rank algorithm)
        sorted_by_rank = sorted(algorithms, key=lambda a: global_ranks_map.get(a, 999.0))
        if reference_algorithm and reference_algorithm in algorithms:
            ref_algo = reference_algorithm
        else:
            ref_algo = sorted_by_rank[0]

        # Decide summary statistic format ("mean_std" vs "median_iqr")
        norm_test_type = test_type.strip().lower()
        if summary_stat in {"mean_std", "median_iqr"}:
            resolved_summary_stat = summary_stat
        elif norm_test_type in {"wilcoxon", "mann_whitney", "kruskal", "friedman"}:
            resolved_summary_stat = "median_iqr"
        else:
            resolved_summary_stat = "mean_std"

        # 2. Evaluate each Instance Row (Omnibus test + Pairwise vs Reference + Descriptive & Normality)
        instance_rows: list[BenchmarkTableInstanceRow] = []
        wins_vs_ref: dict[str, int] = {a: 0 for a in algorithms}
        ties_vs_ref: dict[str, int] = {a: 0 for a in algorithms}
        losses_vs_ref: dict[str, int] = {a: 0 for a in algorithms}
        best_counts: dict[str, int] = {a: 0 for a in algorithms}
        second_best_counts: dict[str, int] = {a: 0 for a in algorithms}
        resolved_omnibus_names: list[str] = []
        resolved_pairwise_names: list[str] = []

        for inst in instances:
            inst_df = sub_df[sub_df["instance"] == inst]
            paired_cell = is_paired_data(inst_df)
            groups: dict[str, np.ndarray] = {}
            for alg in algorithms:
                vals = inst_df[inst_df["algorithm"] == alg]["value"].dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    groups[alg] = vals

            if len(groups) < 2:
                continue

            # Run Omnibus test on this instance
            omnibus_res = _run_omnibus_for_groups(groups, test_type=test_type, alpha=alpha, is_paired=paired_cell)
            resolved_omnibus_names.append(omnibus_res.test_name)

            # Normality & Levene checks
            norm_list = NormalityChecker(alpha=alpha).check(groups)
            norm_map = {a_name: norm_res for a_name, norm_res in zip(groups.keys(), norm_list, strict=False)}
            levene_res = HomogeneityChecker(alpha=alpha).check(groups)
            norm_pass_cnt = sum(1 for n_r in norm_list if n_r.passed)

            # Rank algorithms within this instance by mean or median
            central_scores: dict[str, float] = {}
            for alg, arr in groups.items():
                score = float(np.median(arr)) if resolved_summary_stat == "median_iqr" else float(np.mean(arr))
                central_scores[alg] = score

            ordered_inst_algos = sorted(
                groups.keys(),
                key=lambda a: central_scores[a],
                reverse=maximize,
            )
            best_alg = ordered_inst_algos[0]
            second_best_alg = ordered_inst_algos[1] if len(ordered_inst_algos) > 1 else ""
            best_counts[best_alg] = best_counts.get(best_alg, 0) + 1
            if second_best_alg:
                second_best_counts[second_best_alg] = second_best_counts.get(second_best_alg, 0) + 1

            # Fast vectorized average fractional ranks across runs within this instance
            inst_avg_ranks: dict[str, float] = {}
            group_alg_list = list(groups.keys())
            lengths = {len(groups[a]) for a in group_alg_list}
            if len(lengths) == 1 and next(iter(lengths)) >= 1:
                mat = np.column_stack([groups[a] for a in group_alg_list])
                ranked = stats.rankdata(-mat if maximize else mat, axis=1, method="average")
                mean_ranks = np.mean(ranked, axis=0)
                for idx_a, alg_i in enumerate(group_alg_list):
                    inst_avg_ranks[alg_i] = float(mean_ranks[idx_a])
            else:
                for rank_i, alg_i in enumerate(ordered_inst_algos):
                    inst_avg_ranks[alg_i] = float(rank_i + 1)

            # Pairwise tests against Reference Algorithm on this instance
            raw_pair_p: dict[str, float] = {}
            pair_results: dict[str, StatisticalResult] = {}
            ref_vals = groups.get(ref_algo)
            non_ref_algos = [a for a in groups.keys() if a != ref_algo]

            if ref_vals is not None and len(ref_vals) >= 2:
                for alg in non_ref_algos:
                    alg_vals = groups[alg]
                    if len(alg_vals) >= 2:
                        p_res = _run_pairwise_test(
                            x_algo=alg_vals,
                            x_ref=ref_vals,
                            algo_name=alg,
                            ref_name=ref_algo,
                            test_type=test_type,
                            alpha=alpha,
                            is_paired=paired_cell,
                        )
                        pair_results[alg] = p_res
                        raw_pair_p[alg] = p_res.p_value
                        resolved_pairwise_names.append(p_res.test_name)

            adj_p_list = adjust_pvalues(
                [raw_pair_p[a] for a in non_ref_algos if a in raw_pair_p],
                method=posthoc_correction,
            )
            adj_pair_p: dict[str, float] = {}
            for a_name, p_adj_val in zip([a for a in non_ref_algos if a in raw_pair_p], adj_p_list, strict=False):
                adj_pair_p[a_name] = p_adj_val

            row_cells: dict[str, BenchmarkTableCell] = {}
            for alg, arr in groups.items():
                n_obs = len(arr)
                m_val = float(np.mean(arr))
                s_val = float(np.std(arr, ddof=1)) if n_obs > 1 else 0.0
                med_val = float(np.median(arr))
                q75, q25 = np.percentile(arr, [75, 25]) if n_obs > 0 else (0.0, 0.0)
                iqr_val = float(q75 - q25)
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                skew_v = float(stats.skew(arr)) if n_obs >= 3 and s_val > 0 else 0.0
                kurt_v = float(stats.kurtosis(arr)) if n_obs >= 4 and s_val > 0 else 0.0

                n_check = norm_map.get(alg)
                norm_ok = bool(n_check.passed) if n_check else True
                shapiro_w = float(n_check.statistic) if (n_check and math.isfinite(n_check.statistic)) else None
                shapiro_p = float(n_check.p_value) if (n_check and math.isfinite(n_check.p_value)) else None

                is_ref = alg == ref_algo
                sig_sym = "ref" if is_ref else "≈"
                p_test_name = ""
                p_stat: float | None = None
                p_raw: float | None = None
                p_adj: float | None = None
                eff_name: str | None = None
                eff_val: float | None = None
                eff_mag: str | None = None
                a12_val: float | None = None
                cliff_val: float | None = None
                cohen_val: float | None = None

                if not is_ref and alg in pair_results and ref_vals is not None:
                    pr = pair_results[alg]
                    p_test_name = pr.test_name
                    p_stat = pr.statistic
                    p_raw = pr.p_value
                    p_adj = adj_pair_p.get(alg, pr.p_value)
                    if pr.effect_size:
                        eff_name = pr.effect_size.name
                        eff_val = pr.effect_size.value
                        eff_mag = pr.effect_size.magnitude

                    a12_v, _ = compute_vargha_delaney_a12(arr, ref_vals, maximize=maximize)
                    cliff_v, _ = compute_cliffs_delta(arr, ref_vals, maximize=maximize)
                    cohen_v = _compute_cohens_d_raw(arr, ref_vals)
                    a12_val = a12_v
                    cliff_val = cliff_v
                    cohen_val = cohen_v

                    # Check significance using adjusted p-value
                    effective_p = p_adj if p_adj is not None else p_raw
                    if effective_p is not None and effective_p < alpha:
                        alg_central = med_val if resolved_summary_stat == "median_iqr" else m_val
                        ref_central = (
                            float(np.median(ref_vals))
                            if resolved_summary_stat == "median_iqr"
                            else float(np.mean(ref_vals))
                        )
                        alg_is_better = (alg_central > ref_central) if maximize else (alg_central < ref_central)
                        if alg_is_better:
                            sig_sym = "+"
                            wins_vs_ref[alg] += 1
                        else:
                            sig_sym = "−"
                            losses_vs_ref[alg] += 1
                    else:
                        sig_sym = "≈"
                        ties_vs_ref[alg] += 1

                if resolved_summary_stat == "median_iqr":
                    base_str = f"{_format_num(med_val, precision)} ({_format_num(iqr_val, precision)})"
                else:
                    base_str = f"{_format_num(m_val, precision)} ± {_format_num(s_val, precision)}"

                sym_suffix = "" if is_ref else f" {sig_sym}"
                formatted_cell = f"{base_str}{sym_suffix}"

                row_cells[alg] = BenchmarkTableCell(
                    instance=inst,
                    algorithm=alg,
                    n=n_obs,
                    mean=m_val,
                    std=s_val,
                    median=med_val,
                    iqr=iqr_val,
                    min_val=min_v,
                    max_val=max_v,
                    avg_rank=inst_avg_ranks.get(alg, 1.0),
                    formatted_value=formatted_cell,
                    is_best=(alg == best_alg),
                    is_second_best=(alg == second_best_alg),
                    is_reference=is_ref,
                    significance_symbol=sig_sym,
                    pairwise_test_name=p_test_name,
                    pairwise_statistic=p_stat,
                    pairwise_p_value=p_raw,
                    pairwise_p_adjusted=p_adj,
                    effect_size_name=eff_name,
                    effect_size_value=eff_val,
                    effect_size_magnitude=eff_mag,
                    a12=a12_val,
                    cliffs_delta=cliff_val,
                    cohens_d=cohen_val,
                    normality_passed=norm_ok,
                    shapiro_stat=shapiro_w,
                    shapiro_p_value=shapiro_p,
                    skewness=skew_v,
                    kurtosis=kurt_v,
                    raw_values=[round(float(x), 6) for x in arr.tolist()],
                )

            df_str = (
                f"({omnibus_res.degrees_of_freedom[0]:.0f}, {omnibus_res.degrees_of_freedom[1]:.1f})"
                if isinstance(omnibus_res.degrees_of_freedom, tuple)
                else f"({omnibus_res.degrees_of_freedom})"
                if omnibus_res.degrees_of_freedom is not None
                else ""
            )

            instance_rows.append(
                BenchmarkTableInstanceRow(
                    instance=inst,
                    best_algorithm=best_alg,
                    second_best_algorithm=second_best_alg,
                    omnibus_test_name=omnibus_res.test_name,
                    omnibus_statistic=omnibus_res.statistic,
                    omnibus_p_value=omnibus_res.p_value,
                    omnibus_df=df_str,
                    omnibus_significant=omnibus_res.is_significant,
                    omnibus_effect_size_name=omnibus_res.effect_size.name if omnibus_res.effect_size else "",
                    omnibus_effect_size_value=omnibus_res.effect_size.value if omnibus_res.effect_size else 0.0,
                    omnibus_effect_size_magnitude=omnibus_res.effect_size.magnitude if omnibus_res.effect_size else "",
                    levene_statistic=levene_res.statistic if math.isfinite(levene_res.statistic) else 0.0,
                    levene_p_value=levene_res.p_value if math.isfinite(levene_res.p_value) else 1.0,
                    levene_passed=levene_res.passed,
                    normality_passed_count=norm_pass_cnt,
                    total_groups=len(groups),
                    interpretation=omnibus_res.interpretation,
                    cells=row_cells,
                )
            )

        # 3. Compute Global Omnibus & Post-Hoc vs Reference across Instances
        n_inst = max(1, len(instance_rows))
        k_alg = len(algorithms)
        se_friedman = math.sqrt((k_alg * (k_alg + 1.0)) / (6.0 * n_inst))
        ref_rank = global_ranks_map.get(ref_algo, 1.0)

        non_ref_list = [a for a in algorithms if a != ref_algo]
        friedman_raw_p_map: dict[str, float] = {}
        friedman_z_map: dict[str, float] = {}

        for alg in non_ref_list:
            r_alg = global_ranks_map.get(alg, 1.0)
            z_val = abs(r_alg - ref_rank) / se_friedman if se_friedman > 0 else 0.0
            p_unadj = float(2.0 * (1.0 - stats.norm.cdf(z_val)))
            friedman_z_map[alg] = z_val
            friedman_raw_p_map[alg] = p_unadj

        raw_f_ps = [friedman_raw_p_map[a] for a in non_ref_list]
        holm_f_ps = adjust_pvalues(raw_f_ps, method="holm")
        bonf_f_ps = adjust_pvalues(raw_f_ps, method="bonferroni")
        hoch_f_ps = adjust_pvalues(raw_f_ps, method="hochberg")

        f_holm_map = dict(zip(non_ref_list, holm_f_ps, strict=False))
        f_bonf_map = dict(zip(non_ref_list, bonf_f_ps, strict=False))
        f_hoch_map = dict(zip(non_ref_list, hoch_f_ps, strict=False))

        # Global Wilcoxon Signed-Rank across instance means vs Reference
        inst_means_matrix: dict[str, np.ndarray] = {}
        for alg in algorithms:
            vals_list = [r.cells[alg].mean for r in instance_rows if alg in r.cells]
            inst_means_matrix[alg] = np.array(vals_list, dtype=float)

        global_wilcoxon_stat_map: dict[str, float] = {}
        global_wilcoxon_p_map: dict[str, float] = {}
        ref_inst_means = inst_means_matrix.get(ref_algo, np.array([]))
        for alg in non_ref_list:
            alg_inst_means = inst_means_matrix.get(alg, np.array([]))
            if len(alg_inst_means) >= 2 and len(alg_inst_means) == len(ref_inst_means):
                try:
                    w_res = WilcoxonSignedRankTest().run({alg: alg_inst_means, ref_algo: ref_inst_means}, alpha=alpha)
                    global_wilcoxon_stat_map[alg] = w_res.statistic
                    global_wilcoxon_p_map[alg] = w_res.p_value
                except Exception:
                    pass

        w_only_algos = [a for a in non_ref_list if a in global_wilcoxon_p_map]
        w_holm_ps = adjust_pvalues([global_wilcoxon_p_map[a] for a in w_only_algos], method="holm")
        global_wilcoxon_holm_map = dict(zip(w_only_algos, w_holm_ps, strict=False))

        # Global ANOVA / Welch / Kruskal across instance means (if >= 2 instances)
        global_anova_f: float | None = None
        global_anova_p: float | None = None
        global_welch_f: float | None = None
        global_welch_p: float | None = None
        global_kruskal_h: float | None = None
        global_kruskal_p: float | None = None

        if len(instance_rows) >= 2 and all(len(v) >= 2 for v in inst_means_matrix.values()):
            try:
                ga = OneWayANOVA().run(inst_means_matrix, alpha=alpha)
                global_anova_f, global_anova_p = ga.statistic, ga.p_value
            except Exception:
                pass
            try:
                gw = WelchANOVA().run(inst_means_matrix, alpha=alpha)
                global_welch_f, global_welch_p = gw.statistic, gw.p_value
            except Exception:
                pass
            try:
                gk = KruskalWallisTest().run(inst_means_matrix, alpha=alpha)
                global_kruskal_h, global_kruskal_p = gk.statistic, gk.p_value
            except Exception:
                pass
        elif len(instance_rows) == 1:
            r0 = instance_rows[0]
            if "Welch" in r0.omnibus_test_name:
                global_welch_f, global_welch_p = r0.omnibus_statistic, r0.omnibus_p_value
            elif "ANOVA" in r0.omnibus_test_name:
                global_anova_f, global_anova_p = r0.omnibus_statistic, r0.omnibus_p_value
            elif "Kruskal" in r0.omnibus_test_name:
                global_kruskal_h, global_kruskal_p = r0.omnibus_statistic, r0.omnibus_p_value

        bayes_best_map, bayes_better_map, bayes_rope_map = _fast_bayesian_summary_probs(
            inst_means_matrix=inst_means_matrix,
            algorithms=algorithms,
            ref_algo=ref_algo,
            maximize=maximize,
        )

        rank_order_algos = sorted(algorithms, key=lambda a: global_ranks_map.get(a, 999.0))
        algorithm_summaries: list[AlgorithmSummaryFooter] = []
        for alg in algorithms:
            arr_means = inst_means_matrix.get(alg, np.array([0.0]))
            algorithm_summaries.append(
                AlgorithmSummaryFooter(
                    algorithm=alg,
                    avg_rank=global_ranks_map.get(alg, 1.0),
                    rank_position=rank_order_algos.index(alg) + 1,
                    mean_across_instances=float(np.mean(arr_means)) if len(arr_means) > 0 else 0.0,
                    median_across_instances=float(np.median(arr_means)) if len(arr_means) > 0 else 0.0,
                    wins=wins_vs_ref.get(alg, 0),
                    ties=ties_vs_ref.get(alg, 0),
                    losses=losses_vs_ref.get(alg, 0),
                    best_count=best_counts.get(alg, 0),
                    second_best_count=second_best_counts.get(alg, 0),
                    friedman_z_vs_ref=friedman_z_map.get(alg),
                    friedman_p_unadjusted=friedman_raw_p_map.get(alg),
                    friedman_p_holm=f_holm_map.get(alg),
                    friedman_p_bonferroni=f_bonf_map.get(alg),
                    friedman_p_hochberg=f_hoch_map.get(alg),
                    global_wilcoxon_stat=global_wilcoxon_stat_map.get(alg),
                    global_wilcoxon_p=global_wilcoxon_p_map.get(alg),
                    global_wilcoxon_p_holm=global_wilcoxon_holm_map.get(alg),
                    bayesian_prob_best=bayes_best_map.get(alg),
                    bayesian_prob_better_than_ref=bayes_better_map.get(alg),
                    bayesian_prob_rope_with_ref=bayes_rope_map.get(alg),
                )
            )

        # 4. Build k x k Pairwise Matrix ONLY when requested (table_type == "pairwise_matrix")
        if table_type == "pairwise_matrix":
            pairwise_matrix = self._build_pairwise_matrix(
                sub_df=sub_df,
                target_metric=target_metric,
                algorithms=algorithms,
                instance_rows=instance_rows,
                global_ranks_map=global_ranks_map,
                test_type=test_type,
                posthoc_correction=posthoc_correction,
                maximize=maximize,
                alpha=alpha,
            )
        else:
            pairwise_matrix = []

        most_common_omnibus = (
            max(set(resolved_omnibus_names), key=resolved_omnibus_names.count)
            if resolved_omnibus_names
            else "Friedman Omnibus Test"
        )
        most_common_pairwise = (
            max(set(resolved_pairwise_names), key=resolved_pairwise_names.count)
            if resolved_pairwise_names
            else "Wilcoxon Signed-Rank Test"
        )

        opt_dir_str = "Maximization" if maximize else "Minimization"
        stat_label_str = "Median (IQR)" if resolved_summary_stat == "median_iqr" else "Mean ± SD"
        caption = (
            f"Benchmark evaluation on metric {target_metric} ({opt_dir_str}, {stat_label_str}) across "
            f"{len(instance_rows)} instance(s) and {len(algorithms)} algorithms at α = {alpha}. "
            f"Omnibus test: {most_common_omnibus}; pairwise test: {most_common_pairwise} "
            f"({posthoc_correction.upper()} correction) vs reference algorithm '{ref_algo}' "
            f"(+: significantly better, −: significantly worse, ≈: tie)."
        )

        preliminary_result = StatisticalTableResult(
            table_type=table_type,
            metric=target_metric,
            maximize=maximize,
            alpha=alpha,
            test_type=test_type,
            resolved_omnibus_test=most_common_omnibus,
            resolved_pairwise_test=most_common_pairwise,
            posthoc_correction=posthoc_correction,
            summary_stat=resolved_summary_stat,
            scope=effective_scope,
            selected_instance=effective_inst,
            reference_algorithm=ref_algo,
            algorithms=algorithms,
            instances=[r.instance for r in instance_rows],
            friedman_stat=friedman_stat,
            friedman_p_value=friedman_p,
            iman_davenport_f=iman_f,
            iman_davenport_p_value=iman_p,
            kendalls_w=kendalls_w,
            nemenyi_cd=nemenyi_cd,
            global_anova_f=global_anova_f,
            global_anova_p=global_anova_p,
            global_welch_f=global_welch_f,
            global_welch_p=global_welch_p,
            global_kruskal_h=global_kruskal_h,
            global_kruskal_p=global_kruskal_p,
            instance_rows=instance_rows,
            algorithm_summaries=algorithm_summaries,
            pairwise_matrix=pairwise_matrix,
            caption=caption,
        )

        latex_str = render_latex_table(
            preliminary_result,
            precision=precision,
            document_mode=latex_document_mode,
            notation=latex_notation,
        )
        md_str = render_markdown_table(preliminary_result, precision=precision)

        return preliminary_result.model_copy(
            update={
                "latex_document_mode": latex_document_mode,
                "latex_notation": latex_notation,
                "latex_preview": latex_str,
                "markdown_preview": md_str,
            }
        )

    def _build_pairwise_matrix(
        self,
        sub_df: pd.DataFrame,
        target_metric: str,
        algorithms: list[str],
        instance_rows: list[BenchmarkTableInstanceRow],
        global_ranks_map: dict[str, float],
        test_type: str,
        posthoc_correction: str,
        maximize: bool,
        alpha: float,
    ) -> list[list[PairwiseMatrixCell]]:
        """Construct the full k x k pairwise comparison matrix across all algorithm pairs."""
        _ = (sub_df, target_metric)
        k = len(algorithms)
        n_inst = len(instance_rows)
        norm_type = test_type.strip().lower()

        # Collect upper-triangle raw p-values for family-wise multiple testing adjustment
        pair_keys: list[tuple[int, int]] = []
        raw_p_list: list[float] = []
        raw_stat_map: dict[tuple[int, int], tuple[str, float, float]] = {}
        inst_pair_p_cache: dict[tuple[int, int, str], float | None] = {}

        for i in range(k):
            for j in range(i + 1, k):
                alg_a = algorithms[i]
                alg_b = algorithms[j]
                pair_keys.append((i, j))

                for r_inst in instance_rows:
                    if alg_a not in r_inst.cells or alg_b not in r_inst.cells:
                        continue
                    arr_a = np.array(r_inst.cells[alg_a].raw_values, dtype=float)
                    arr_b = np.array(r_inst.cells[alg_b].raw_values, dtype=float)
                    inst_p_val: float | None = None
                    if len(arr_a) >= 2 and len(arr_b) >= 2:
                        try:
                            inst_pr = _run_pairwise_test(
                                x_algo=arr_a,
                                x_ref=arr_b,
                                algo_name=alg_a,
                                ref_name=alg_b,
                                test_type=test_type,
                                alpha=alpha,
                                is_paired=(len(arr_a) == len(arr_b)),
                            )
                            inst_p_val = inst_pr.p_value
                        except Exception:
                            inst_p_val = None
                    inst_pair_p_cache[(i, j, r_inst.instance)] = inst_p_val

                if n_inst > 1:
                    # Multi-instance comparison across instance means (or medians)
                    vals_a = np.array(
                        [r.cells[alg_a].mean for r in instance_rows if alg_a in r.cells and alg_b in r.cells],
                        dtype=float,
                    )
                    vals_b = np.array(
                        [r.cells[alg_b].mean for r in instance_rows if alg_a in r.cells and alg_b in r.cells],
                        dtype=float,
                    )
                    if norm_type == "friedman":
                        se_f = math.sqrt((k * (k + 1.0)) / (6.0 * max(1, n_inst)))
                        z_f = abs(global_ranks_map.get(alg_a, 1.0) - global_ranks_map.get(alg_b, 1.0)) / se_f
                        p_f = float(2.0 * (1.0 - stats.norm.cdf(z_f)))
                        raw_stat_map[(i, j)] = ("Friedman Post-Hoc z-Test", z_f, p_f)
                        raw_p_list.append(p_f)
                    elif len(vals_a) >= 2:
                        p_res = _run_pairwise_test(
                            x_algo=vals_a,
                            x_ref=vals_b,
                            algo_name=alg_a,
                            ref_name=alg_b,
                            test_type="wilcoxon" if norm_type == "auto" else test_type,
                            alpha=alpha,
                            is_paired=True,
                        )
                        raw_stat_map[(i, j)] = (p_res.test_name, p_res.statistic, p_res.p_value)
                        raw_p_list.append(p_res.p_value)
                    else:
                        raw_stat_map[(i, j)] = ("Wilcoxon Signed-Rank Test", 0.0, 1.0)
                        raw_p_list.append(1.0)
                else:
                    # Single-instance comparison across runs
                    r0 = instance_rows[0] if instance_rows else None
                    vals_a = np.array(r0.cells[alg_a].raw_values if (r0 and alg_a in r0.cells) else [], dtype=float)
                    vals_b = np.array(r0.cells[alg_b].raw_values if (r0 and alg_b in r0.cells) else [], dtype=float)
                    if len(vals_a) >= 2 and len(vals_b) >= 2:
                        p_res = _run_pairwise_test(
                            x_algo=vals_a,
                            x_ref=vals_b,
                            algo_name=alg_a,
                            ref_name=alg_b,
                            test_type=test_type,
                            alpha=alpha,
                            is_paired=(len(vals_a) == len(vals_b)),
                        )
                        raw_stat_map[(i, j)] = (p_res.test_name, p_res.statistic, p_res.p_value)
                        raw_p_list.append(p_res.p_value)
                    else:
                        raw_stat_map[(i, j)] = ("Mann-Whitney U Test", 0.0, 1.0)
                        raw_p_list.append(1.0)

        adj_p_list = adjust_pvalues(raw_p_list, method=posthoc_correction)
        adj_p_map: dict[tuple[int, int], float] = dict(zip(pair_keys, adj_p_list, strict=False))

        matrix: list[list[PairwiseMatrixCell]] = []
        for i in range(k):
            row_cells: list[PairwiseMatrixCell] = []
            alg_row = algorithms[i]
            for j in range(k):
                alg_col = algorithms[j]
                if i == j:
                    row_cells.append(
                        PairwiseMatrixCell(
                            algo_row=alg_row,
                            algo_col=alg_col,
                            test_name="Self",
                            statistic=0.0,
                            p_value=1.0,
                            p_adjusted=1.0,
                            is_significant=False,
                            direction="self",
                            symbol="—",
                            interpretation=f"Diagonal self-comparison cell for {alg_row}.",
                        )
                    )
                    continue

                pair_key = (min(i, j), max(i, j))
                t_name, stat_val, p_raw = raw_stat_map.get(pair_key, ("Pairwise Test", 0.0, 1.0))
                p_adj = adj_p_map.get(pair_key, p_raw)

                # Build per-instance breakdown between alg_row and alg_col
                breakdown: list[PairwiseInstanceComparison] = []
                wins_r = 0
                ties_r = 0
                losses_r = 0
                sig_wins_r = 0
                sig_ties_r = 0
                sig_losses_r = 0
                inst_sym_chars: list[str] = []
                row_agg_vals: list[float] = []
                col_agg_vals: list[float] = []

                for r_inst in instance_rows:
                    if alg_row not in r_inst.cells or alg_col not in r_inst.cells:
                        continue
                    c_r = r_inst.cells[alg_row]
                    c_c = r_inst.cells[alg_col]
                    row_agg_vals.append(c_r.mean)
                    col_agg_vals.append(c_c.mean)

                    arr_r = np.array(c_r.raw_values, dtype=float)
                    arr_c = np.array(c_c.raw_values, dtype=float)
                    inst_a12, _ = compute_vargha_delaney_a12(arr_r, arr_c, maximize=maximize)
                    inst_p = inst_pair_p_cache.get((pair_key[0], pair_key[1], r_inst.instance))

                    diff_m = (c_r.mean - c_c.mean) if maximize else (c_c.mean - c_r.mean)
                    if abs(diff_m) <= 1e-12:
                        winner_str = "tie"
                        ties_r += 1
                    elif diff_m > 0:
                        winner_str = "row"
                        wins_r += 1
                    else:
                        winner_str = "col"
                        losses_r += 1

                    inst_sig = bool(inst_p is not None and inst_p < alpha and abs(diff_m) > 1e-12)
                    if inst_sig:
                        if diff_m > 0:
                            inst_sym = "+"
                            sig_wins_r += 1
                        else:
                            inst_sym = "-"
                            sig_losses_r += 1
                    else:
                        inst_sym = "="
                        sig_ties_r += 1
                    inst_sym_chars.append(inst_sym)

                    breakdown.append(
                        PairwiseInstanceComparison(
                            instance=r_inst.instance,
                            mean_row=c_r.mean,
                            std_row=c_r.std,
                            median_row=c_r.median,
                            iqr_row=c_r.iqr,
                            mean_col=c_c.mean,
                            std_col=c_c.std,
                            median_col=c_c.median,
                            iqr_col=c_c.iqr,
                            winner=winner_str,
                            symbol=inst_sym,
                            is_significant=inst_sig,
                            p_value=inst_p,
                            a12=inst_a12,
                        )
                    )

                instance_symbols_str = "".join(inst_sym_chars)

                if n_inst == 1 and instance_rows:
                    c_r0 = instance_rows[0].cells[alg_row]
                    c_c0 = instance_rows[0].cells[alg_col]
                    vec_r = np.array(c_r0.raw_values, dtype=float)
                    vec_c = np.array(c_c0.raw_values, dtype=float)
                else:
                    vec_r = np.array(row_agg_vals, dtype=float)
                    vec_c = np.array(col_agg_vals, dtype=float)

                a12_pair, a12_mag = compute_vargha_delaney_a12(vec_r, vec_c, maximize=maximize)
                cliff_pair, _ = compute_cliffs_delta(vec_r, vec_c, maximize=maximize)
                cohen_pair = _compute_cohens_d_raw(vec_r, vec_c)

                mean_r = float(np.mean(vec_r)) if len(vec_r) > 0 else 0.0
                mean_c = float(np.mean(vec_c)) if len(vec_c) > 0 else 0.0
                med_r = float(np.median(vec_r)) if len(vec_r) > 0 else 0.0
                med_c = float(np.median(vec_c)) if len(vec_c) > 0 else 0.0

                is_sig = p_adj < alpha
                row_is_better = (mean_r > mean_c) if maximize else (mean_r < mean_c)
                if is_sig:
                    direction = "better" if row_is_better else "worse"
                    sym = "+" if row_is_better else "−"
                else:
                    direction = "tie"
                    sym = "≈"

                prob_r, prob_rope_v, prob_c = _fast_bayesian_pair_probs(
                    vec_r=vec_r,
                    vec_c=vec_c,
                    maximize=maximize,
                )

                interp = (
                    f"{alg_row} vs {alg_col} ({t_name}): statistic = {stat_val:.4f}, "
                    f"raw p = {_format_p(p_raw)}, {posthoc_correction.upper()} adjusted p = {_format_p(p_adj)} "
                    f"({'Significant' if is_sig else 'Not significant'} at α = {alpha}). "
                    f"Vargha-Delaney A12 = {a12_pair:.3f} ({a12_mag}), Cliff's δ = {cliff_pair:.3f}. "
                    f"Statistical instance record (+/=/-): +{sig_wins_r} / ={sig_ties_r} / -{sig_losses_r} "
                    f"({instance_symbols_str})."
                )

                row_cells.append(
                    PairwiseMatrixCell(
                        algo_row=alg_row,
                        algo_col=alg_col,
                        test_name=t_name,
                        statistic=stat_val,
                        p_value=p_raw,
                        p_adjusted=p_adj,
                        is_significant=is_sig,
                        direction=direction,
                        symbol=sym,
                        instance_symbols=instance_symbols_str,
                        sig_wins=sig_wins_r,
                        sig_ties=sig_ties_r,
                        sig_losses=sig_losses_r,
                        wins=wins_r,
                        ties=ties_r,
                        losses=losses_r,
                        mean_row=mean_r,
                        mean_col=mean_c,
                        median_row=med_r,
                        median_col=med_c,
                        effect_size_name="Vargha-Delaney A12",
                        effect_size_value=a12_pair,
                        effect_size_magnitude=a12_mag,
                        a12=a12_pair,
                        cliffs_delta=cliff_pair,
                        cohens_d=cohen_pair,
                        prob_row_better=prob_r,
                        prob_rope=prob_rope_v,
                        prob_col_better=prob_c,
                        interpretation=interp,
                        instance_breakdown=breakdown,
                    )
                )
            matrix.append(row_cells)
        return matrix


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in identifiers and text, converting Unicode symbols."""
    if not text:
        return ""
    out = str(text)

    structural = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]
    for char, esc in structural:
        out = out.replace(char, esc)

    symbols = [
        ("±", r"$\pm$"),
        ("α", r"$\alpha$"),
        ("β", r"$\beta$"),
        ("γ", r"$\gamma$"),
        ("δ", r"$\delta$"),
        ("ε", r"$\varepsilon$"),
        ("χ", r"$\chi$"),
        ("−", r"$-$"),  # U+2212 unicode minus sign
        ("≈", r"$\approx$"),  # U+2248 almost equal
        ("—", r"---"),  # U+2014 em dash
        ("–", r"--"),  # U+2013 en dash
        ("²", r"$^2$"),
        ("³", r"$^3$"),
        ("≤", r"$\le$"),
        ("≥", r"$\ge$"),
        ("≠", r"$\ne$"),
        ("∞", r"$\infty$"),
    ]
    for sym, tex in symbols:
        out = out.replace(sym, tex)

    return out


def _build_standalone_preamble(title: str = "Algorithms Comparison") -> list[str]:
    """Construct preamble for a standalone compilable LaTeX document."""
    return [
        r"\documentclass{article}",
        f"\\title{{{_escape_latex(title)}}}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{colortbl}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{float}",
        r"\usepackage{rotating}",
        r"\usepackage{tabularx}",
        r"\usepackage[margin=1in]{geometry}",
        r"\IfFileExists{siunitx.sty}{",
        r"  \usepackage{siunitx}",
        r"  \sisetup{output-exponent-marker=\text{e}}",
        r"}{}",
        r"\xdefinecolor{gray95}{gray}{0.65}",
        r"\xdefinecolor{gray25}{gray}{0.8}",
        r"\author{SAES Benchmark Suite}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Tables}",
    ]


def _build_standalone_postamble() -> list[str]:
    """Construct postamble for a standalone compilable LaTeX document."""
    return [
        r"\end{document}",
    ]


def _build_snippet_header() -> list[str]:
    """Construct comment header for snippet inclusion in external papers."""
    return [
        r"% Generated by SAES (Stochastic Algorithm Evaluation Suite)",
        r"% Required packages: \usepackage[table]{xcolor}, \usepackage{colortbl}, \usepackage{booktabs}",
        r"% Required math packages: \usepackage{amsmath}",
        r"% Color definitions: \xdefinecolor{gray95}{gray}{0.65} (best)",
        r"%                    \xdefinecolor{gray25}{gray}{0.8} (second-best)",
    ]


def _format_latex_cell_value(
    val: float,
    std_or_iqr: float,
    notation: str = "scientific",
    is_iqr: bool = False,
    precision: int = 4,
) -> str:
    """Format numeric values for LaTeX according to the requested notation style."""
    if not math.isfinite(val) or not math.isfinite(std_or_iqr):
        return "---"

    if notation == "siunitx":
        return f"\\SI{{{val:.2e}}}{{}}_{{ \\SI{{{std_or_iqr:.2e}}}{{}} }}"
    elif notation == "scientific":
        return f"{val:.2e}_{{ {std_or_iqr:.2e} }}"
    else:  # standard decimal
        if is_iqr:
            return f"{_format_num(val, precision)} \\; ({_format_num(std_or_iqr, precision)})"
        return f"{_format_num(val, precision)} \\pm {_format_num(std_or_iqr, precision)}"


def render_latex_table(
    result: StatisticalTableResult,
    precision: int = 4,
    document_mode: str = "standalone",
    notation: str = "scientific",
) -> str:
    """Render a publication-grade LaTeX table from StatisticalTableResult.

    Parameters:
        result: The StatisticalTableResult to render.
        precision: Decimal digits for standard notation.
        document_mode: 'standalone' (complete compilable document with preamble) or 'snippet' (table float only).
        notation: 'scientific' ($2.55e-02_{...}$), 'siunitx' (\\SI{...}), or 'standard' ($0.0255 \\pm 0.0033$).
    """
    algos = result.algorithms
    is_standalone = document_mode.strip().lower() == "standalone"

    if result.table_type == "pairwise_matrix":
        col_algos = algos[1:] if len(algos) > 1 else algos
        row_algos_count = max(1, len(algos) - 1)
        col_spec = "l|" + "|".join(["c"] * len(col_algos))
        inst_list_str = "[ " + ",  ".join(f"'{_escape_latex(inst)}'" for inst in result.instances) + "]"
        test_label = _escape_latex(result.resolved_pairwise_test.replace("Signed-Rank ", ""))
        header_cols = " & ".join(_escape_latex(a) for a in col_algos)

        lines_pw: list[str] = [
            r"\begin{table*}[htbp]",
            (
                rf"\caption{{{_escape_latex(result.metric)}.  {test_label} 1vs1 Table. "
                r"Each symbol in the cells represents a problem. "
                r"Symbol + indicates that the row algorithm performs better with statistical confidence; "
                r"symbol - indicates that the row algorithm performs worse with statistical confidence;  "
                r"symbol = implies that the differences are not significant. "
                rf"Instances (in order) : {inst_list_str} (Length: {len(result.instances)})}}"
            ),
            r"\vspace{1mm}",
            r"\centering",
            r"\begin{scriptsize}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            f"& {header_cols} \\\\ \\hline",
        ]

        for r_idx in range(row_algos_count):
            if r_idx >= len(result.pairwise_matrix):
                break
            row_cells_pw = result.pairwise_matrix[r_idx]
            r_name = _escape_latex(algos[r_idx])
            cell_tex_list: list[str] = []
            for c_idx in range(1, len(algos)):
                if c_idx < r_idx:
                    cell_tex_list.append(r"\texttt{}")
                elif c_idx == r_idx:
                    cell_tex_list.append("")
                else:
                    cell_obj = row_cells_pw[c_idx]
                    sym_seq = cell_obj.instance_symbols or (
                        "+" if cell_obj.symbol == "+" else "-" if cell_obj.symbol in ("−", "-") else "="
                    )
                    cell_tex_list.append(rf"\texttt{{{sym_seq}}}")
            lines_pw.append(f"{r_name} & {' & '.join(cell_tex_list)} \\\\")

        lines_pw.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{scriptsize}",
                r"\end{table*}",
            ]
        )
        if is_standalone:
            return "\n".join(
                _build_standalone_preamble(title=f"Algorithms Comparison 1vs1: {result.metric}")
                + lines_pw
                + _build_standalone_postamble()
            )
        return "\n".join(_build_snippet_header() + lines_pw)

    if result.table_type == "omnibus_posthoc":
        lines_posthoc: list[str] = [
            r"\begin{table*}[htbp]",
            rf"\caption{{{_escape_latex(result.caption)}}}",
            rf"\label{{tab:saes_{result.metric.lower()}_{result.table_type}}}",
            r"\vspace{1mm}",
            r"\centering",
            r"\begin{scriptsize}",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{l|c|c|c|c|c|c|c}",
            r"\hline",
            (
                r"\textbf{Algorithm} & \textbf{Rank} & \textbf{Mean} & \textbf{$z$-stat} & "
                r"\textbf{$p_{\text{unadj}}$} & \textbf{$p_{\text{Holm}}$} & "
                r"\textbf{$p_{\text{Bonf}}$} & \textbf{Win/Tie/Loss ($+/ \approx / -$)} \\ \hline"
            ),
        ]
        sorted_sums = sorted(result.algorithm_summaries, key=lambda s: s.avg_rank)
        min_rank = min((x.avg_rank for x in sorted_sums), default=1.0)
        for s in sorted_sums:
            is_ref = s.algorithm == result.reference_algorithm
            name_tex = f"\\textbf{{{_escape_latex(s.algorithm)}}} (Ref)" if is_ref else _escape_latex(s.algorithm)
            z_str = f"{s.friedman_z_vs_ref:.3f}" if s.friedman_z_vs_ref is not None else "---"
            pu_str = _format_p(s.friedman_p_unadjusted).replace("—", "---")
            ph_str = _format_p(s.friedman_p_holm).replace("—", "---")
            pb_str = _format_p(s.friedman_p_bonferroni).replace("—", "---")
            wtl_str = "Control" if is_ref else f"{s.wins} / {s.ties} / {s.losses}"
            mean_str = (
                f"{s.mean_across_instances:.2e}"
                if notation in ("scientific", "siunitx")
                else _format_num(s.mean_across_instances, precision)
            )
            shading = r"\cellcolor{gray95}" if s.avg_rank == min_rank else ""
            lines_posthoc.append(
                f"{name_tex} & {shading}{s.avg_rank:.2f} ({s.rank_position}) & "
                f"${mean_str}$ & {z_str} & {pu_str} & {ph_str} & {pb_str} & {wtl_str} \\\\"
            )
        lines_posthoc.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{scriptsize}",
                r"\end{table*}",
            ]
        )
        if is_standalone:
            return "\n".join(
                _build_standalone_preamble(title=f"Friedman & Post-Hoc Ranking: {result.metric}")
                + lines_posthoc
                + _build_standalone_postamble()
            )
        return "\n".join(_build_snippet_header() + lines_posthoc)

    # Default: instance_algorithm table (optimized borders and spacing)
    col_spec = "l|" + "|".join(["c"] * len(algos)) + "|c"
    tab_size = r"\begin{scriptsize}" if len(algos) >= 5 else r"\begin{footnotesize}"
    tab_colsep = r"\setlength{\tabcolsep}{3.5pt}" if len(algos) >= 6 else r"\setlength{\tabcolsep}{5pt}"

    lines_tab: list[str] = [
        r"\begin{table*}[htbp]",
        rf"\caption{{{_escape_latex(result.caption)}}}",
        rf"\label{{tab:saes_{result.metric.lower()}_{result.table_type}}}",
        r"\vspace{1mm}",
        r"\centering",
        tab_size,
        tab_colsep,
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
    ]
    header_cols = " & ".join(
        f"\\textbf{{{_escape_latex(a)}}}" + (" (Ref)" if a == result.reference_algorithm else "") for a in algos
    )
    lines_tab.append(f"& {header_cols} & \\textbf{{Omnibus ($p$-value)}} \\\\ \\hline")

    for r in result.instance_rows:
        c_list: list[str] = []
        for a in algos:
            c = r.cells.get(a)
            if not c or not math.isfinite(c.mean) or not math.isfinite(c.std):
                c_list.append("---")
                continue

            is_iqr = result.summary_stat == "median_iqr"
            center_v = c.median if is_iqr else c.mean
            spread_v = c.iqr if is_iqr else c.std
            cell_tex = _format_latex_cell_value(
                center_v,
                spread_v,
                notation=notation,
                is_iqr=is_iqr,
                precision=precision,
            )

            shading = ""
            if c.is_best:
                shading = r"\cellcolor{gray95}"
            elif c.is_second_best:
                shading = r"\cellcolor{gray25}"

            sym_tex = ""
            if not c.is_reference:
                if c.significance_symbol == "+":
                    sym_tex = " +" if notation in ("scientific", "siunitx") else " $^+$"
                elif c.significance_symbol in ("−", "-"):
                    sym_tex = " -" if notation in ("scientific", "siunitx") else " $^-$"
                elif c.significance_symbol in ("≈", "=", "~"):
                    sym_tex = " =" if notation in ("scientific", "siunitx") else " $^\\approx$"

            c_list.append(f"{shading}${cell_tex}${sym_tex}")

        om_p_str = _format_p(r.omnibus_p_value).replace("—", "---")
        om_str = f"{r.omnibus_statistic:.2f} ({om_p_str})"
        lines_tab.append(f"{_escape_latex(r.instance)} & {' & '.join(c_list)} & {om_str} \\\\")

    lines_tab.append(r"\hline")
    wtl_cells: list[str] = []
    rank_cells: list[str] = []
    sum_map = {s.algorithm: s for s in result.algorithm_summaries}
    for a in algos:
        s_item = sum_map.get(a)
        if s_item:
            if a == result.reference_algorithm:
                wtl_cells.append("Reference")
            else:
                wtl_cells.append(f"\\textbf{{{s_item.wins}}} / \\textbf{{{s_item.ties}}} / \\textbf{{{s_item.losses}}}")
            rank_cells.append(f"{s_item.avg_rank:.2f} ({s_item.rank_position})")
        else:
            wtl_cells.append("---")
            rank_cells.append("---")

    cd_tex = f"CD = {result.nemenyi_cd or 0:.3f}"
    lines_tab.append(f"\\textbf{{$+ / \\approx / -$ vs Ref}} & {' & '.join(wtl_cells)} & {cd_tex} \\\\")
    f_p_str = _format_p(result.friedman_p_value).replace("—", "---")
    f_summary = (
        f"$\\chi^2_F={result.friedman_stat:.2f}$ ($p={f_p_str}$)"
        if result.friedman_stat is not None
        else "---"
    )
    lines_tab.append(f"\\textbf{{Avg. Rank}} & {' & '.join(rank_cells)} & {f_summary} \\\\ \\hline")
    lines_tab.extend(
        [
            r"\end{tabular}",
            r"\end{scriptsize}" if len(algos) >= 5 else r"\end{footnotesize}",
            r"\end{table*}",
        ]
    )

    if is_standalone:
        title_metric = f"SAES Benchmark: {result.metric}"
        return "\n".join(
            _build_standalone_preamble(title=title_metric)
            + lines_tab
            + _build_standalone_postamble()
        )
    return "\n".join(_build_snippet_header() + lines_tab)


def render_markdown_table(result: StatisticalTableResult, precision: int = 4) -> str:
    """Render a clean GitHub-Flavored Markdown table from StatisticalTableResult."""
    algos = result.algorithms
    lines: list[str] = []
    lines.append(f"### {result.caption}\n")

    if result.table_type == "pairwise_matrix":
        col_algos = algos[1:] if len(algos) > 1 else algos
        row_algos_count = max(1, len(algos) - 1)
        headers = ["Algorithm"] + col_algos
        lines.append(f"**Instances (in order):** `{', '.join(result.instances)}`\n")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r_idx in range(row_algos_count):
            if r_idx >= len(result.pairwise_matrix):
                break
            row = result.pairwise_matrix[r_idx]
            r_cells = [f"**{algos[r_idx]}**"]
            for c_idx in range(1, len(algos)):
                if c_idx <= r_idx:
                    r_cells.append("")
                else:
                    c = row[c_idx]
                    r_cells.append(f"`{c.instance_symbols}` (+{c.sig_wins}/={c.sig_ties}/-{c.sig_losses})")
            lines.append("| " + " | ".join(r_cells) + " |")
        return "\n".join(lines)

    headers = (
        ["Instance"] + [f"{a} (Ref)" if a == result.reference_algorithm else a for a in algos] + ["Omnibus (p-value)"]
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in result.instance_rows:
        row_vals = [f"**{r.instance}**"]
        for a in algos:
            inst_cell = r.cells.get(a)
            if not inst_cell:
                row_vals.append("—")
                continue
            txt = f"**{inst_cell.formatted_value}**" if inst_cell.is_best else inst_cell.formatted_value
            row_vals.append(txt)
        row_vals.append(f"{r.omnibus_statistic:.3f} (p={_format_p(r.omnibus_p_value)})")
        lines.append("| " + " | ".join(row_vals) + " |")

    sum_map = {s.algorithm: s for s in result.algorithm_summaries}
    rank_row = (
        ["**Avg. Friedman Rank**"]
        + [f"{sum_map[a].avg_rank:.2f} (#{sum_map[a].rank_position})" if a in sum_map else "—" for a in algos]
        + [
            f"χ²={result.friedman_stat:.2f} (p={_format_p(result.friedman_p_value)})"
            if result.friedman_stat is not None
            else "—"
        ]
    )
    wtl_row = (
        ["**+ / ≈ / − vs Ref**"]
        + [
            "Reference"
            if a == result.reference_algorithm
            else f"{sum_map[a].wins} / {sum_map[a].ties} / {sum_map[a].losses}"
            if a in sum_map
            else "—"
            for a in algos
        ]
        + [f"CD={result.nemenyi_cd or 0.0:.3f}"]
    )

    lines.append("| " + " | ".join(rank_row) + " |")
    lines.append("| " + " | ".join(wtl_row) + " |")
    return "\n".join(lines)


def _extract_flat_headers_and_rows(
    result: StatisticalTableResult,
) -> tuple[list[str], list[list[str]], list[list[bool]]]:
    """Extract headers, 2D string rows, and 2D bold flags for CSV, DOCX, HTML, and PDF exports."""
    algos = result.algorithms
    if result.table_type == "pairwise_matrix":
        col_algos = algos[1:] if len(algos) > 1 else algos
        row_algos_count = max(1, len(algos) - 1)
        headers = [""] + col_algos
        rows: list[list[str]] = []
        bolds: list[list[bool]] = []
        for r_idx in range(row_algos_count):
            if r_idx >= len(result.pairwise_matrix):
                break
            r_list = result.pairwise_matrix[r_idx]
            r_str = [algos[r_idx]]
            b_row = [True]
            for c_idx in range(1, len(algos)):
                if c_idx <= r_idx:
                    r_str.append("")
                    b_row.append(False)
                else:
                    c = r_list[c_idx]
                    r_str.append(c.instance_symbols or f"p={_format_p(c.p_adjusted)} ({c.symbol})")
                    b_row.append(c.is_significant and c.direction == "better")
            rows.append(r_str)
            bolds.append(b_row)
        return headers, rows, bolds

    if result.table_type == "omnibus_posthoc":
        headers = [
            "Algorithm",
            "Avg Rank",
            "Rank #",
            "Mean",
            "Median",
            "Friedman z",
            "p (Unadj)",
            "p (Holm)",
            "p (Bonf)",
            "+ / ≈ / − vs Ref",
            "Bayesian P(Best)",
        ]
        rows = []
        bolds = []
        for s in sorted(result.algorithm_summaries, key=lambda x: x.avg_rank):
            is_best = s.rank_position == 1
            is_ref = s.algorithm == result.reference_algorithm
            rows.append(
                [
                    f"{s.algorithm} (Ref)" if is_ref else s.algorithm,
                    f"{s.avg_rank:.3f}",
                    str(s.rank_position),
                    _format_num(s.mean_across_instances, 4),
                    _format_num(s.median_across_instances, 4),
                    f"{s.friedman_z_vs_ref:.3f}" if s.friedman_z_vs_ref is not None else "—",
                    _format_p(s.friedman_p_unadjusted),
                    _format_p(s.friedman_p_holm),
                    _format_p(s.friedman_p_bonferroni),
                    "Reference" if is_ref else f"{s.wins} / {s.ties} / {s.losses}",
                    f"{(s.bayesian_prob_best or 0.0) * 100:.1f}%" if s.bayesian_prob_best is not None else "—",
                ]
            )
            bolds.append([is_best] * len(headers))
        return headers, rows, bolds

    # Default: instance_algorithm
    headers = (
        ["Instance"]
        + [f"{a} (Ref)" if a == result.reference_algorithm else a for a in algos]
        + ["Omnibus Statistic", "Omnibus p-value"]
    )
    rows = []
    bolds = []
    for r in result.instance_rows:
        r_str = [r.instance]
        b_row = [False]
        for a in algos:
            inst_cell = r.cells.get(a)
            if not inst_cell:
                r_str.append("—")
                b_row.append(False)
            else:
                r_str.append(inst_cell.formatted_value)
                b_row.append(inst_cell.is_best)
        r_str.append(f"{r.omnibus_statistic:.3f}")
        r_str.append(_format_p(r.omnibus_p_value))
        b_row.extend([False, r.omnibus_significant])
        rows.append(r_str)
        bolds.append(b_row)

    sum_map = {s.algorithm: s for s in result.algorithm_summaries}
    rank_r = ["Avg. Friedman Rank"]
    wtl_r = ["+ / ≈ / − vs Ref"]
    for a in algos:
        s_foot = sum_map.get(a)
        rank_r.append(f"{s_foot.avg_rank:.2f} (#{s_foot.rank_position})" if s_foot else "—")
        wtl_cell = (
            "Reference"
            if a == result.reference_algorithm
            else (f"{s_foot.wins} / {s_foot.ties} / {s_foot.losses}" if s_foot else "—")
        )
        wtl_r.append(wtl_cell)
    rank_r.extend(
        [
            f"χ² = {result.friedman_stat:.2f}" if result.friedman_stat is not None else "—",
            _format_p(result.friedman_p_value),
        ]
    )
    wtl_r.extend([f"CD = {result.nemenyi_cd or 0.0:.3f}", f"α = {result.alpha}"])
    rows.append(rank_r)
    bolds.append([True] * len(headers))
    rows.append(wtl_r)
    bolds.append([True] * len(headers))
    return headers, rows, bolds


def _build_docx_bytes(result: StatisticalTableResult) -> bytes:
    """Create a valid OpenXML Word (.docx) document with an academic formatted table."""
    headers, rows, bolds = _extract_flat_headers_and_rows(result)
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ET.register_namespace("w", w_ns)

    def w_tag(tag: str) -> str:
        return f"{{{w_ns}}}{tag}"

    doc = ET.Element(w_tag("document"))
    body = ET.SubElement(doc, w_tag("body"))

    # Title paragraph
    p_title = ET.SubElement(body, w_tag("p"))
    r_title = ET.SubElement(p_title, w_tag("r"))
    rpr_title = ET.SubElement(r_title, w_tag("rPr"))
    ET.SubElement(rpr_title, w_tag("b"))
    sz_title = ET.SubElement(rpr_title, w_tag("sz"))
    sz_title.set(w_tag("val"), "26")
    t_title = ET.SubElement(r_title, w_tag("t"))
    t_title.text = f"SAES Statistical Benchmark Table — Metric: {result.metric}"

    # Caption paragraph
    p_cap = ET.SubElement(body, w_tag("p"))
    r_cap = ET.SubElement(p_cap, w_tag("r"))
    rpr_cap = ET.SubElement(r_cap, w_tag("rPr"))
    ET.SubElement(rpr_cap, w_tag("i"))
    sz_cap = ET.SubElement(rpr_cap, w_tag("sz"))
    sz_cap.set(w_tag("val"), "18")
    t_cap = ET.SubElement(r_cap, w_tag("t"))
    t_cap.text = result.caption

    # Table element
    tbl = ET.SubElement(body, w_tag("tbl"))
    tbl_pr = ET.SubElement(tbl, w_tag("tblPr"))
    tbl_borders = ET.SubElement(tbl_pr, w_tag("tblBorders"))
    for border_name in ["top", "bottom", "insideH"]:
        b_el = ET.SubElement(tbl_borders, w_tag(border_name))
        b_el.set(w_tag("val"), "single")
        b_el.set(w_tag("sz"), "8" if border_name in {"top", "bottom"} else "4")
        b_el.set(w_tag("space"), "0")
        b_el.set(w_tag("color"), "1E293B")

    # Header row
    tr_head = ET.SubElement(tbl, w_tag("tr"))
    for h_text in headers:
        tc = ET.SubElement(tr_head, w_tag("tc"))
        p = ET.SubElement(tc, w_tag("p"))
        r = ET.SubElement(p, w_tag("r"))
        rpr = ET.SubElement(r, w_tag("rPr"))
        ET.SubElement(rpr, w_tag("b"))
        sz = ET.SubElement(rpr, w_tag("sz"))
        sz.set(w_tag("val"), "18")
        t = ET.SubElement(r, w_tag("t"))
        t.text = h_text

    # Data rows
    for row_vals, bold_flags in zip(rows, bolds, strict=False):
        tr = ET.SubElement(tbl, w_tag("tr"))
        for cell_txt, is_b in zip(row_vals, bold_flags, strict=False):
            tc = ET.SubElement(tr, w_tag("tc"))
            p = ET.SubElement(tc, w_tag("p"))
            r = ET.SubElement(p, w_tag("r"))
            rpr = ET.SubElement(r, w_tag("rPr"))
            if is_b:
                ET.SubElement(rpr, w_tag("b"))
            sz = ET.SubElement(rpr, w_tag("sz"))
            sz.set(w_tag("val"), "18")
            t = ET.SubElement(r, w_tag("t"))
            t.text = cell_txt

    xml_bytes = ET.tostring(doc, encoding="utf-8", xml_declaration=True)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", xml_bytes)
    return buf.getvalue()


PDF_THEMES = {
    "slate": {
        "header_bg": "#1e293b",
        "header_fg": "#ffffff",
        "best_bg": "#ecfdf5",
        "best_fg": "#065f46",
        "zebra_even": "#ffffff",
        "zebra_odd": "#f8fafc",
        "border": "#cbd5e1",
        "footer_bg": "#f1f5f9",
        "footer_fg": "#0f172a",
        "text": "#1e293b",
        "title_color": "#0f172a",
        "caption_color": "#475569",
    },
    "academic_bw": {
        "header_bg": "#f8fafc",
        "header_fg": "#000000",
        "best_bg": "#f1f5f9",
        "best_fg": "#000000",
        "zebra_even": "#ffffff",
        "zebra_odd": "#ffffff",
        "border": "#000000",
        "footer_bg": "#ffffff",
        "footer_fg": "#000000",
        "text": "#000000",
        "title_color": "#000000",
        "caption_color": "#334155",
    },
    "emerald": {
        "header_bg": "#064e3b",
        "header_fg": "#ffffff",
        "best_bg": "#d1fae5",
        "best_fg": "#065f46",
        "zebra_even": "#ffffff",
        "zebra_odd": "#f0fdf4",
        "border": "#a7f3d0",
        "footer_bg": "#ecfdf5",
        "footer_fg": "#064e3b",
        "text": "#0f172a",
        "title_color": "#064e3b",
        "caption_color": "#047857",
    },
    "navy": {
        "header_bg": "#1e3a8a",
        "header_fg": "#ffffff",
        "best_bg": "#dbeafe",
        "best_fg": "#1e40af",
        "zebra_even": "#ffffff",
        "zebra_odd": "#f8fafc",
        "border": "#bfdbfe",
        "footer_bg": "#eff6ff",
        "footer_fg": "#1e3a8a",
        "text": "#0f172a",
        "title_color": "#1e3a8a",
        "caption_color": "#1d4ed8",
    },
    "crimson": {
        "header_bg": "#881337",
        "header_fg": "#ffffff",
        "best_bg": "#ffe4e6",
        "best_fg": "#9f1239",
        "zebra_even": "#ffffff",
        "zebra_odd": "#fff1f2",
        "border": "#fecdd3",
        "footer_bg": "#ffe4e6",
        "footer_fg": "#881337",
        "text": "#0f172a",
        "title_color": "#881337",
        "caption_color": "#be123c",
    },
}


def _build_pdf_bytes(
    result: StatisticalTableResult,
    title: str | None = None,
    caption: str | None = None,
    theme: str = "slate",
    orientation: str = "landscape",
    font_size: float = 8.5,
    highlight_best: bool = True,
    show_zebra: bool = True,
    grid_style: str = "full",
) -> bytes:
    """Render a publication-grade vector PDF table using Matplotlib PDF backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers, rows, bolds = _extract_flat_headers_and_rows(result)
    n_rows = len(rows)
    n_cols = max(1, len(headers))

    palette = PDF_THEMES.get((theme or "").lower().strip(), PDF_THEMES["slate"])

    is_portrait = (orientation or "").lower().strip() == "portrait"
    if is_portrait:
        fig_w = max(8.5, min(16.0, 1.8 * n_cols))
        fig_h = max(11.0, min(32.0, 2.0 + 0.42 * (n_rows + 2)))
    else:
        fig_w = max(11.0, min(24.0, 2.2 * n_cols))
        fig_h = max(5.0, min(24.0, 1.8 + 0.42 * (n_rows + 2)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    title_str = (
        title.strip()
        if title and title.strip()
        else f"SAES Statistical Benchmark Table — Metric: {result.metric} (α = {result.alpha})"
    )
    caption_str = caption.strip() if caption and caption.strip() else result.caption

    ax.set_title(
        title_str,
        fontsize=max(10.0, font_size + 3.0),
        fontweight="bold",
        loc="left",
        pad=14,
        color=palette["title_color"],
    )

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.35 if font_size < 8.0 else (1.45 if font_size < 10.0 else 1.6))

    border_color = palette["border"]
    clean_grid = (grid_style or "").lower().strip()

    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if clean_grid == "horizontal":
            cell.set_edgecolor(border_color)
            cell.visible_edges = "TB"
        elif clean_grid == "minimal":
            if r_idx == 0:
                cell.visible_edges = "TB"
                cell.set_linewidth(1.2)
            elif r_idx == n_rows:
                cell.visible_edges = "B"
                cell.set_linewidth(1.2)
            else:
                cell.visible_edges = ""
            cell.set_edgecolor(border_color)
        else:
            cell.set_edgecolor(border_color)

        if r_idx == 0:
            cell.set_facecolor(palette["header_bg"])
            cell.set_text_props(color=palette["header_fg"], weight="bold")
        else:
            data_r = r_idx - 1
            is_b = bolds[data_r][c_idx] if (data_r < len(bolds) and c_idx < len(bolds[data_r])) else False
            is_footer = result.table_type == "instance_algorithm" and data_r >= len(result.instance_rows)
            if is_footer:
                cell.set_facecolor(palette["footer_bg"])
                cell.set_text_props(weight="bold", color=palette["footer_fg"])
            elif is_b and highlight_best:
                cell.set_facecolor(palette["best_bg"])
                cell.set_text_props(weight="bold", color=palette["best_fg"])
            else:
                bg = palette["zebra_odd"] if (show_zebra and data_r % 2 != 0) else palette["zebra_even"]
                cell.set_facecolor(bg)
                cell.set_text_props(color=palette["text"])

    fig.text(
        0.02,
        0.02,
        caption_str,
        fontsize=max(7.0, font_size - 1.0),
        color=palette["caption_color"],
        style="italic",
        wrap=True,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return buf.getvalue()


def export_statistical_table(
    result: StatisticalTableResult,
    export_format: str = "latex",
    document_mode: str | None = None,
    notation: str | None = None,
    title: str | None = None,
    caption: str | None = None,
    pdf_theme: str = "slate",
    pdf_orientation: str = "landscape",
    pdf_font_size: float = 8.5,
    pdf_highlight_best: bool = True,
    pdf_show_zebra: bool = True,
    pdf_grid_style: str = "full",
) -> tuple[bytes, str, str]:
    """Export a StatisticalTableResult into the specified file format.

    Supported export_format values:
        - 'latex' / 'tex'
        - 'csv'
        - 'markdown' / 'md'
        - 'html'
        - 'docx' / 'word'
        - 'pdf'

    Returns:
        Tuple of (file_bytes, media_type, extension)
    """
    fmt = export_format.strip().lower()
    if fmt in {"latex", "tex"}:
        doc_mode = document_mode or getattr(result, "latex_document_mode", "standalone") or "standalone"
        not_mode = notation or getattr(result, "latex_notation", "scientific") or "scientific"
        tex = render_latex_table(result, document_mode=doc_mode, notation=not_mode)
        return tex.encode("utf-8"), "application/x-tex", "tex"

    if fmt in {"markdown", "md"}:
        md = result.markdown_preview or render_markdown_table(result)
        return md.encode("utf-8"), "text/markdown", "md"

    headers, rows, bolds = _extract_flat_headers_and_rows(result)

    if fmt == "csv":
        out = io.StringIO()
        df_out = pd.DataFrame(rows, columns=headers)
        df_out.to_csv(out, index=False)
        return out.getvalue().encode("utf-8"), "text/csv", "csv"

    effective_title = (
        title.strip() if title and title.strip() else f"SAES Statistical Benchmark Table — {result.metric}"
    )
    effective_caption = caption.strip() if caption and caption.strip() else result.caption

    if fmt == "html":
        html_lines = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<style>body{font-family:system-ui,sans-serif;padding:24px;color:#0f172a}",
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}",
            "th{background:#0f172a;color:#fff;padding:8px 12px;text-align:center}",
            "td{border-bottom:1px solid #cbd5e1;padding:8px 12px;text-align:center}",
            "tr:nth-child(even) td{background:#f8fafc}",
            ".best{font-weight:700;color:#047857;background:#ecfdf5!important}",
            "</style></head><body>",
            f"<h2>{effective_title}</h2>",
            f"<p><em>{effective_caption}</em></p>",
            "<table><thead><tr>",
        ]
        for h in headers:
            html_lines.append(f"<th>{h}</th>")
        html_lines.append("</tr></thead><tbody>")
        for r_vals, b_vals in zip(rows, bolds, strict=False):
            html_lines.append("<tr>")
            for val, is_b in zip(r_vals, b_vals, strict=False):
                cls = " class='best'" if is_b else ""
                html_lines.append(f"<td{cls}>{val}</td>")
            html_lines.append("</tr>")
        html_lines.append("</tbody></table></body></html>")
        return "\n".join(html_lines).encode("utf-8"), "text/html", "html"

    if fmt in {"docx", "word"}:
        return (
            _build_docx_bytes(result),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        )

    if fmt == "pdf":
        return (
            _build_pdf_bytes(
                result,
                title=title,
                caption=caption,
                theme=pdf_theme,
                orientation=pdf_orientation,
                font_size=pdf_font_size,
                highlight_best=pdf_highlight_best,
                show_zebra=pdf_show_zebra,
                grid_style=pdf_grid_style,
            ),
            "application/pdf",
            "pdf",
        )

    raise StatisticalAnalysisError(f"Unsupported table export format: '{export_format}'.")


def export_flat_table(
    headers: list[str],
    rows: list[list[str]],
    bolds: list[list[bool]],
    title: str,
    caption: str,
    export_format: str = "csv",
    document_mode: str = "standalone",
) -> tuple[bytes, str, str]:
    """Export any tabular grid (headers, rows, bold flags) into PDF, DOCX, LaTeX, CSV, Markdown, or HTML."""
    fmt = export_format.strip().lower()

    if fmt == "csv":
        out = io.StringIO()
        df_out = pd.DataFrame(rows, columns=headers)
        df_out.to_csv(out, index=False)
        return out.getvalue().encode("utf-8"), "text/csv", "csv"

    if fmt in {"markdown", "md"}:
        lines = [
            f"### {title}",
            "",
            f"> {caption}",
            "",
            "| " + " | ".join(headers) + " |",
            "| :-- " + "| :--: " * max(0, len(headers) - 1) + "|",
        ]
        for r_vals, b_vals in zip(rows, bolds, strict=False):
            cells_fmt = [f"**{v}**" if is_b else v for v, is_b in zip(r_vals, b_vals, strict=False)]
            lines.append("| " + " | ".join(cells_fmt) + " |")
        return "\n".join(lines).encode("utf-8"), "text/markdown", "md"

    if fmt in {"latex", "tex"}:
        col_spec = "l|" + "|".join(["c"] * max(0, len(headers) - 1))
        esc_headers = [f"\\textbf{{{_escape_latex(h)}}}" for h in headers]
        lines_flat = [
            r"\begin{table*}[htbp]",
            r"\vspace{1mm}",
            r"\centering",
            rf"\caption{{{_escape_latex(caption)}}}",
            r"\begin{scriptsize}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            " & ".join(esc_headers) + r" \\ \hline",
        ]
        for r_vals, b_vals in zip(rows, bolds, strict=False):
            c_strs = []
            for v, is_b in zip(r_vals, b_vals, strict=False):
                esc_v = _escape_latex(v)
                c_strs.append(rf"\cellcolor{{gray95}}\textbf{{{esc_v}}}" if is_b else esc_v)
            lines_flat.append(" & ".join(c_strs) + r" \\")
        lines_flat.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{scriptsize}",
                r"\end{table*}",
            ]
        )
        if document_mode.strip().lower() == "standalone":
            tex_str = "\n".join(
                _build_standalone_preamble(title=title)
                + lines_flat
                + _build_standalone_postamble()
            )
        else:
            tex_str = "\n".join(_build_snippet_header() + lines_flat)
        return tex_str.encode("utf-8"), "application/x-tex", "tex"

    if fmt == "html":
        html_lines = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<style>body{font-family:system-ui,sans-serif;padding:24px;color:#0f172a}",
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}",
            "th{background:#0f172a;color:#fff;padding:8px 12px;text-align:center}",
            "td{border-bottom:1px solid #cbd5e1;padding:8px 12px;text-align:center}",
            "tr:nth-child(even) td{background:#f8fafc}",
            ".best{font-weight:700;color:#047857;background:#ecfdf5!important}",
            "</style></head><body>",
            f"<h2>{title}</h2>",
            f"<p><em>{caption}</em></p>",
            "<table><thead><tr>",
        ]
        for h in headers:
            html_lines.append(f"<th>{h}</th>")
        html_lines.append("</tr></thead><tbody>")
        for r_vals, b_vals in zip(rows, bolds, strict=False):
            html_lines.append("<tr>")
            for val, is_b in zip(r_vals, b_vals, strict=False):
                cls = " class='best'" if is_b else ""
                html_lines.append(f"<td{cls}>{val}</td>")
            html_lines.append("</tr>")
        html_lines.append("</tbody></table></body></html>")
        return "\n".join(html_lines).encode("utf-8"), "text/html", "html"

    if fmt in {"docx", "word"}:
        w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        ET.register_namespace("w", w_ns)

        def w_tag(tag: str) -> str:
            return f"{{{w_ns}}}{tag}"

        doc = ET.Element(w_tag("document"))
        body = ET.SubElement(doc, w_tag("body"))

        p_title = ET.SubElement(body, w_tag("p"))
        r_title = ET.SubElement(p_title, w_tag("r"))
        rpr_title = ET.SubElement(r_title, w_tag("rPr"))
        ET.SubElement(rpr_title, w_tag("b"))
        sz_title = ET.SubElement(rpr_title, w_tag("sz"))
        sz_title.set(w_tag("val"), "26")
        t_title = ET.SubElement(r_title, w_tag("t"))
        t_title.text = title

        p_cap = ET.SubElement(body, w_tag("p"))
        r_cap = ET.SubElement(p_cap, w_tag("r"))
        rpr_cap = ET.SubElement(r_cap, w_tag("rPr"))
        ET.SubElement(rpr_cap, w_tag("i"))
        sz_cap = ET.SubElement(rpr_cap, w_tag("sz"))
        sz_cap.set(w_tag("val"), "18")
        t_cap = ET.SubElement(r_cap, w_tag("t"))
        t_cap.text = caption

        tbl = ET.SubElement(body, w_tag("tbl"))
        tbl_pr = ET.SubElement(tbl, w_tag("tblPr"))
        tbl_borders = ET.SubElement(tbl_pr, w_tag("tblBorders"))
        for border_name in ["top", "bottom", "insideH"]:
            b_el = ET.SubElement(tbl_borders, w_tag(border_name))
            b_el.set(w_tag("val"), "single")
            b_el.set(w_tag("sz"), "8" if border_name in {"top", "bottom"} else "4")
            b_el.set(w_tag("space"), "0")
            b_el.set(w_tag("color"), "1E293B")

        tr_head = ET.SubElement(tbl, w_tag("tr"))
        for h_text in headers:
            tc = ET.SubElement(tr_head, w_tag("tc"))
            p = ET.SubElement(tc, w_tag("p"))
            r = ET.SubElement(p, w_tag("r"))
            rpr = ET.SubElement(r, w_tag("rPr"))
            ET.SubElement(rpr, w_tag("b"))
            sz = ET.SubElement(rpr, w_tag("sz"))
            sz.set(w_tag("val"), "18")
            t = ET.SubElement(r, w_tag("t"))
            t.text = h_text

        for row_vals, bold_flags in zip(rows, bolds, strict=False):
            tr = ET.SubElement(tbl, w_tag("tr"))
            for cell_txt, is_b in zip(row_vals, bold_flags, strict=False):
                tc = ET.SubElement(tr, w_tag("tc"))
                p = ET.SubElement(tc, w_tag("p"))
                r = ET.SubElement(p, w_tag("r"))
                rpr = ET.SubElement(r, w_tag("rPr"))
                if is_b:
                    ET.SubElement(rpr, w_tag("b"))
                sz = ET.SubElement(rpr, w_tag("sz"))
                sz.set(w_tag("val"), "18")
                t = ET.SubElement(r, w_tag("t"))
                t.text = cell_txt

        xml_bytes = ET.tostring(doc, encoding="utf-8", xml_declaration=True)
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", xml_bytes)
        return (
            buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        )

    if fmt == "pdf":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pdf_rows = rows[:150] if rows else [["No matching rows"] + [""] * max(0, len(headers) - 1)]
        pdf_bolds = bolds[:150] if bolds else [[False] * len(headers)]
        n_rows = len(pdf_rows)
        n_cols = max(1, len(headers))

        fig_w = max(10.0, min(22.0, 2.1 * n_cols))
        fig_h = max(4.0, min(28.0, 1.8 + 0.38 * (n_rows + 2)))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=14, color="#0f172a")

        tbl_pdf = ax.table(
            cellText=pdf_rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tbl_pdf.auto_set_font_size(False)
        tbl_pdf.set_fontsize(8.0)
        tbl_pdf.scale(1.0, 1.4)

        for (r_idx, c_idx), cell in tbl_pdf.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if r_idx == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="white", weight="bold")
            else:
                data_r = r_idx - 1
                is_b = (
                    pdf_bolds[data_r][c_idx] if (data_r < len(pdf_bolds) and c_idx < len(pdf_bolds[data_r])) else False
                )
                if is_b:
                    cell.set_facecolor("#ecfdf5")
                    cell.set_text_props(weight="bold", color="#065f46")
                else:
                    cell.set_facecolor("#ffffff" if data_r % 2 == 0 else "#f8fafc")
                    cell.set_text_props(color="#1e293b")

        fig.text(0.02, 0.02, caption, fontsize=8, color="#475569", style="italic", wrap=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        return buf.getvalue(), "application/pdf", "pdf"

    raise StatisticalAnalysisError(f"Unsupported table export format: '{export_format}'.")
