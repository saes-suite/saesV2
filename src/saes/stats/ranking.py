"""Non-parametric multi-algorithm ranking tests: Friedman + Nemenyi CD & Bayesian Posterior Ranks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    AlgorithmRankItem,
    BayesianPosteriorItem,
    BayesianRankResult,
    CriticalDistanceResult,
)

# Studentized range statistic q_alpha / sqrt(2) critical values for two-tailed Nemenyi test
# (Demšar 2006, Table 5a; exact studentized_range fallback via SciPy for arbitrary k and alpha)
_NEMENYI_Q_ALPHA: dict[float, dict[int, float]] = {
    0.01: {
        2: 2.576,
        3: 2.913,
        4: 3.113,
        5: 3.255,
        6: 3.364,
        7: 3.452,
        8: 3.526,
        9: 3.590,
        10: 3.646,
    },
    0.05: {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    },
    0.10: {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    },
}


def _get_nemenyi_q_alpha(k: int, alpha: float) -> float:
    """Return critical value q_alpha for Nemenyi two-tailed test."""
    try:
        # Exact studentized range distribution from scipy.stats: q_alpha = Q(1 - alpha, k, inf) / sqrt(2)
        q_exact = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0))
        if math.isfinite(q_exact) and q_exact > 0:
            return q_exact
    except Exception:
        pass

    alpha_key = 0.01 if alpha <= 0.01 else (0.10 if alpha >= 0.10 else 0.05)
    table = _NEMENYI_Q_ALPHA.get(alpha_key, _NEMENYI_Q_ALPHA[0.05])
    clamped_k = min(10, max(2, k))
    return table[clamped_k]


def _extract_block_matrix(
    df: pd.DataFrame,
    metric: str,
    scope: str = "all-instances",
    instance: str | None = None,
) -> tuple[list[str], np.ndarray, bool, str, list[str], list[str], str, str]:
    """Extract a complete (n_blocks x k_algorithms) matrix of observations for ranking.

    Returns:
        (algorithms, matrix [n_blocks, k], maximize, scope_label,
         metrics_list, instances_list, metrics_label, instances_label)
    """
    all_available_metrics = sorted(df["metric"].unique().tolist())
    if metric.lower() in {"all-metrics", "all", "*"}:
        target_metrics = all_available_metrics
    else:
        target_metrics = [metric]

    metric_df = df[df["metric"].isin(target_metrics)].copy()
    if metric_df.empty:
        raise StatisticalAnalysisError(f"No observations found for metric '{metric}'.")

    is_multi_metric = len(target_metrics) > 1
    maximize = True if is_multi_metric else bool(metric_df["maximize"].iloc[0])
    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    k = len(algorithms)
    if k < 2:
        raise StatisticalAnalysisError(f"Ranking tests require at least 2 algorithms, got {k}.")

    all_instances = sorted(metric_df["instance"].unique().tolist())
    use_single_instance = scope == "single-instance" or (len(all_instances) == 1 and instance is not None)

    if not use_single_instance and len(all_instances) > 1:
        target_instances = all_instances
    else:
        resolved_inst = instance if (instance and instance in all_instances) else all_instances[0]
        target_instances = [resolved_inst]

    rows: list[list[float]] = []
    for m_name in target_metrics:
        m_sub = metric_df[metric_df["metric"] == m_name]
        if m_sub.empty:
            continue
        m_max = bool(m_sub["maximize"].iloc[0])

        if not use_single_instance and len(all_instances) > 1:
            for inst in target_instances:
                inst_df = m_sub[m_sub["instance"] == inst]
                block_vals: list[float] = []
                for algo in algorithms:
                    vals = inst_df[inst_df["algorithm"] == algo]["value"].dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        block_vals.append(float(np.mean(vals)))
                if len(block_vals) == k:
                    if is_multi_metric:
                        arr = np.array(block_vals, dtype=float)
                        oriented = arr if m_max else -arr
                        b_min, b_max = float(np.min(oriented)), float(np.max(oriented))
                        span = max(1e-9, b_max - b_min)
                        normed = 1.0 + (oriented - b_min) / span
                        rows.append(normed.tolist())
                    else:
                        rows.append(block_vals)
        else:
            target_inst = target_instances[0]
            inst_df = m_sub[m_sub["instance"] == target_inst]
            run_ids = sorted(inst_df["run_id"].unique().tolist())
            for run_id in run_ids:
                run_df = inst_df[inst_df["run_id"] == run_id]
                block_vals = []
                for algo in algorithms:
                    vals = run_df[run_df["algorithm"] == algo]["value"].dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        block_vals.append(float(vals[0]))
                if len(block_vals) == k:
                    if is_multi_metric:
                        arr = np.array(block_vals, dtype=float)
                        oriented = arr if m_max else -arr
                        b_min, b_max = float(np.min(oriented)), float(np.max(oriented))
                        span = max(1e-9, b_max - b_min)
                        normed = 1.0 + (oriented - b_min) / span
                        rows.append(normed.tolist())
                    else:
                        rows.append(block_vals)

    if not rows:
        raise StatisticalAnalysisError(f"No complete blocks found across all {k} algorithms for metric '{metric}'.")

    if len(target_metrics) == 1:
        metrics_label = f"Metric: {target_metrics[0]}"
    else:
        metrics_label = f"Metrics ({len(target_metrics)}): {', '.join(target_metrics)}"

    if len(target_instances) == 1:
        instances_label = f"Instance: {target_instances[0]} ({len(rows)} Blocks)"
        scope_label = f"Instance: {target_instances[0]} ({len(rows)} Runs)"
    else:
        inst_str = (
            ", ".join(target_instances)
            if len(target_instances) <= 6
            else f"{', '.join(target_instances[:5])}, …, {target_instances[-1]}"
        )
        instances_label = f"Instances ({len(target_instances)}): {inst_str}"
        scope_label = f"Across {len(target_instances)} Instances ({len(rows)} Blocks)"

    matrix = np.array(rows, dtype=float)
    return (
        algorithms,
        matrix,
        maximize,
        scope_label,
        target_metrics,
        target_instances,
        metrics_label,
        instances_label,
    )


def _compute_fractional_ranks_matrix(matrix: np.ndarray, maximize: bool) -> np.ndarray:
    """Compute fractional (average tie) ranks per row (block). Rank 1 is best."""
    n_blocks, k = matrix.shape
    ranks_matrix = np.zeros((n_blocks, k), dtype=float)
    for b in range(n_blocks):
        row = -matrix[b] if maximize else matrix[b]
        ranks_matrix[b] = stats.rankdata(row, method="average")
    return ranks_matrix


class FriedmanNemenyiTest:
    """Friedman non-parametric omnibus test, Iman-Davenport F, and Nemenyi Critical Distance (CD)."""

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        scope: str = "all-instances",
        instance: str | None = None,
    ) -> CriticalDistanceResult:
        (
            algorithms,
            matrix,
            maximize,
            scope_label,
            metrics_list,
            instances_list,
            metrics_label,
            instances_label,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        ranks_matrix = _compute_fractional_ranks_matrix(matrix, maximize=maximize)

        avg_ranks = np.mean(ranks_matrix, axis=0)
        mean_values = np.mean(matrix, axis=0)

        # Sort algorithms from best (lowest average rank) to worst (highest average rank)
        order = np.argsort(avg_ranks)
        rank_items = [
            AlgorithmRankItem(
                algorithm=algorithms[idx],
                avg_rank=float(avg_ranks[idx]),
                mean_value=float(mean_values[idx]),
            )
            for idx in order
        ]

        # Friedman chi-square statistic and p-value
        sum_sq_ranks = float(np.sum(avg_ranks**2))
        friedman_stat = float(((12.0 * n_blocks) / (k * (k + 1.0))) * (sum_sq_ranks - (k * (k + 1.0) ** 2) / 4.0))
        friedman_stat = max(0.0, friedman_stat)
        friedman_p = float(1.0 - stats.chi2.cdf(friedman_stat, df=k - 1))

        # Iman-Davenport F-statistic
        denom = max(1e-9, n_blocks * (k - 1.0) - friedman_stat)
        iman_f = float(((n_blocks - 1.0) * friedman_stat) / denom) if n_blocks > 1 else 0.0
        iman_p = float(1.0 - stats.f.cdf(iman_f, k - 1, (k - 1) * (n_blocks - 1))) if n_blocks > 1 else 1.0

        # Nemenyi Critical Distance (CD)
        q_alpha = _get_nemenyi_q_alpha(k=k, alpha=alpha)
        cd = float(q_alpha * math.sqrt((k * (k + 1.0)) / (6.0 * n_blocks)))

        # Maximal cliques of non-significantly different algorithms (|R_j - R_i| <= CD)
        raw_cliques: list[tuple[int, int]] = []
        for i in range(len(rank_items)):
            max_j = i
            for j in range(i + 1, len(rank_items)):
                if rank_items[j].avg_rank - rank_items[i].avg_rank <= cd + 1e-9:
                    max_j = j
            if max_j > i:
                is_subset = any(s <= i and e >= max_j for s, e in raw_cliques)
                if not is_subset:
                    raw_cliques.append((i, max_j))

        cliques = [(rank_items[i].avg_rank, rank_items[j].avg_rank) for i, j in raw_cliques]
        p_str = f"{friedman_p:.2e}" if friedman_p < 0.0001 else f"{friedman_p:.4f}"
        params_label = (
            f"α = {alpha}  •  CD = {cd:.3f}  •  k = {k}, N = {n_blocks}  •  χ² = {friedman_stat:.2f} (p = {p_str})"
        )

        return CriticalDistanceResult(
            metric=metric,
            scope=scope,
            scope_label=scope_label,
            metrics_list=metrics_list,
            instances_list=instances_list,
            metrics_label=metrics_label,
            instances_label=instances_label,
            params_label=params_label,
            alpha=alpha,
            k=k,
            n_blocks=n_blocks,
            cd=cd,
            friedman_stat=friedman_stat,
            friedman_p_value=friedman_p,
            iman_davenport_f=iman_f,
            iman_davenport_p_value=iman_p,
            maximize=maximize,
            ranks=rank_items,
            cliques=cliques,
        )


class BayesianRankingTest:
    """Bayesian Sign-Test and Signed-Rank Simplex (ROPE) & Posterior Ranking test (Benavoli et al. 2017)."""

    def run(
        self,
        df: pd.DataFrame,
        metric: str,
        scope: str = "all-instances",
        instance: str | None = None,
        rope_percent: float = 1.0,
        num_samples: int = 1500,
        seed: int = 42,
        algo_a: str | None = None,
        algo_b: str | None = None,
        method: str = "signed-rank",
    ) -> BayesianRankResult:
        is_sign = method.strip().lower() in {"sign", "bayesian-sign", "sign-test"}
        normalized_method = "sign" if is_sign else "signed-rank"
        (
            algorithms,
            matrix,
            maximize,
            scope_label,
            metrics_list,
            instances_list,
            metrics_label,
            instances_label,
        ) = _extract_block_matrix(df=df, metric=metric, scope=scope, instance=instance)
        n_blocks, k = matrix.shape
        ranks_matrix = _compute_fractional_ranks_matrix(matrix, maximize=maximize)

        method_seed_offset = 101 if normalized_method == "sign" else 0
        rng = np.random.default_rng(seed + n_blocks * 31 + k * 17 + method_seed_offset)
        # Draw Dirichlet(1, ..., 1) weights of shape (num_samples, n_blocks)
        weights = rng.dirichlet(np.ones(n_blocks, dtype=float), size=num_samples)

        # Posterior expected ranks: shape (num_samples, k)
        posterior_ranks = weights @ ranks_matrix

        # Count how often each algorithm achieves the best (minimum) expected rank
        best_indices = np.argmin(posterior_ranks, axis=1)
        best_counts = np.bincount(best_indices, minlength=k)

        mean_ranks = np.mean(posterior_ranks, axis=0)
        ref_idx = int(np.argmin(mean_ranks))
        reference_algo = algorithms[ref_idx]

        items: list[BayesianPosteriorItem] = []
        rope_threshold = max(1e-6, rope_percent / 100.0)

        for idx, algo in enumerate(algorithms):
            samples_arr = posterior_ranks[:, idx]
            q025, q25, q50, q75, q975 = np.percentile(samples_arr, [2.5, 25.0, 50.0, 75.0, 97.5])

            # Bayesian Sign-Test with ROPE against the top-ranked reference algorithm
            wins = 0
            rope_ties = 0
            for b in range(n_blocks):
                v_algo = float(matrix[b, idx])
                v_ref = float(matrix[b, ref_idx])
                scale = max(1e-9, (abs(v_algo) + abs(v_ref)) / 2.0)
                rel_diff = ((v_algo - v_ref) if maximize else (v_ref - v_algo)) / scale
                if abs(rel_diff) <= rope_threshold:
                    rope_ties += 1
                elif rel_diff > rope_threshold:
                    wins += 1

            prob_better = 0.0 if idx == ref_idx else float((wins + 0.5) / (n_blocks + 1.5))
            prob_rope = 1.0 if idx == ref_idx else float((rope_ties + 0.5) / (n_blocks + 1.5))

            items.append(
                BayesianPosteriorItem(
                    algorithm=algo,
                    mean_rank=float(mean_ranks[idx]),
                    median_rank=float(q50),
                    ci_lower=float(q025),
                    ci_upper=float(q975),
                    q25=float(q25),
                    q75=float(q75),
                    prob_best=float(best_counts[idx] / num_samples),
                    prob_better_than_ref=prob_better,
                    prob_rope_with_ref=prob_rope,
                    samples=[round(float(x), 4) for x in samples_arr.tolist()],
                )
            )

        items.sort(key=lambda item: item.mean_rank)

        # Resolve pairwise algorithms (algo_a = Left vertex, algo_b = Right vertex) for Bayesian Simplex test
        sorted_algos_by_rank = [item.algorithm for item in items]
        resolved_b = algo_b if (algo_b and algo_b in algorithms) else sorted_algos_by_rank[0]
        if algo_a and algo_a in algorithms and algo_a != resolved_b:
            resolved_a = algo_a
        else:
            fallback_candidates = [a for a in sorted_algos_by_rank if a != resolved_b]
            resolved_a = fallback_candidates[0] if fallback_candidates else sorted_algos_by_rank[0]

        idx_a = algorithms.index(resolved_a)
        idx_b = algorithms.index(resolved_b)
        vals_a = matrix[:, idx_a]
        vals_b = matrix[:, idx_b]

        # Compute paired relative differences where > 0 means algo_a (Left) beats algo_b (Right)
        pair_scales = np.maximum(1e-9, (np.abs(vals_a) + np.abs(vals_b)) / 2.0)
        raw_diffs = (vals_a - vals_b) if maximize else (vals_b - vals_a)
        z_obs = raw_diffs / pair_scales

        # Scale empirical weight so posterior cloud maintains informative dispersion across any n_blocks
        eff_weight = 1.0 if n_blocks <= 12 else (12.0 / float(n_blocks))

        if normalized_method == "sign":
            # Benavoli et al. (2017) Bayesian Sign Test on Simplex:
            # Counts only signs of differences (wins_left, ties_rope, wins_right), invariant to difference magnitude.
            wins_left = float(np.sum(z_obs > rope_threshold)) * eff_weight
            ties_rope = float(np.sum(np.abs(z_obs) <= rope_threshold)) * eff_weight
            wins_right = float(np.sum(z_obs < -rope_threshold)) * eff_weight
            w_sign = rng.dirichlet([wins_left + 0.5, ties_rope + 0.5, wins_right + 0.5], size=num_samples)
            theta_left = w_sign[:, 0]
            theta_rope = w_sign[:, 1]
            theta_right = w_sign[:, 2]
        else:
            # Benavoli et al. (2017) Bayesian Signed-Rank Test on Simplex:
            # Dirichlet Process posterior over pairwise sums (z_i + z_j) vs 2*ROPE, sensitive to difference magnitude.
            spread = max(rope_threshold * 2.5, float(np.std(z_obs)) if len(z_obs) > 1 else rope_threshold * 2.5)
            z_aug = np.concatenate([[-spread, 0.0, spread], z_obs])
            pairwise_sums = z_aug[:, None] + z_aug[None, :]
            two_rope = 2.0 * rope_threshold

            ind_left = (pairwise_sums > two_rope).astype(float)
            ind_rope = (np.abs(pairwise_sums) <= two_rope).astype(float)
            ind_right = (pairwise_sums < -two_rope).astype(float)

            alpha_sr = np.concatenate([[0.5, 0.5, 0.5], np.full(n_blocks, eff_weight, dtype=float)])
            w_sr = rng.dirichlet(alpha_sr, size=num_samples)

            theta_left = np.einsum("mi,ij,mj->m", w_sr, ind_left, w_sr)
            theta_rope = np.einsum("mi,ij,mj->m", w_sr, ind_rope, w_sr)
            theta_right = np.einsum("mi,ij,mj->m", w_sr, ind_right, w_sr)

        theta_sum = np.maximum(1e-9, theta_left + theta_rope + theta_right)
        theta_left = theta_left / theta_sum
        theta_rope = theta_rope / theta_sum
        theta_right = theta_right / theta_sum

        # Decision region probabilities: fraction of posterior samples in each region of the simplex
        theta_stack = np.column_stack([theta_left, theta_rope, theta_right])
        dominant_region = np.argmax(theta_stack, axis=1)
        p_left_val = round(float(np.mean(dominant_region == 0)), 3)
        p_rope_val = round(float(np.mean(dominant_region == 1)), 3)
        p_right_val = round(max(0.0, 1.0 - p_left_val - p_rope_val), 3)

        # Barycentric to 2D Cartesian Equilateral Simplex projection:
        # Left vertex (algo_a) = (0, 0), Right vertex (algo_b) = (1, 0), Top vertex (ROPE) = (0.5, sqrt(3)/2)
        h_tri = math.sqrt(3.0) / 2.0
        x_coords = theta_right + 0.5 * theta_rope
        y_coords = h_tri * theta_rope

        # Compute 2D Gaussian KDE density for color-coding points from dark navy (periphery) to cyan/white (core)
        try:
            kde = stats.gaussian_kde(np.vstack([x_coords, y_coords]))
            raw_density = kde(np.vstack([x_coords, y_coords]))
        except Exception:
            dist_from_center = np.hypot(x_coords - np.mean(x_coords), y_coords - np.mean(y_coords))
            raw_density = np.exp(-dist_from_center / max(1e-6, float(np.std(dist_from_center))))

        d_min = float(np.min(raw_density))
        d_max = float(np.max(raw_density))
        norm_density = (raw_density - d_min) / max(1e-9, d_max - d_min)

        # Sort points by ascending density so highest-density points render on top
        density_order = np.argsort(norm_density)
        x_sorted = x_coords[density_order]
        y_sorted = y_coords[density_order]
        d_sorted = norm_density[density_order]
        p_left_sorted = theta_left[density_order]
        p_rope_sorted = theta_rope[density_order]
        p_right_sorted = theta_right[density_order]

        # Pre-computed equilateral triangle border and internal ROPE decision boundary segments
        cx, cy = 0.5, round(h_tri / 3.0, 6)
        half_h = round(h_tri / 2.0, 6)
        top_h = round(h_tri, 6)
        triangle_x = [0.0, 1.0, 0.5, 0.0]
        triangle_y = [0.0, 0.0, top_h, 0.0]
        boundary_lines: list[tuple[float, float, float, float]] = [
            (cx, cy, 0.5, 0.0),
            (cx, cy, 0.25, half_h),
            (cx, cy, 0.75, half_h),
        ]

        method_tag = "Bayesian Sign" if normalized_method == "sign" else "Bayesian Signed-Rank"
        params_label = (
            f"{method_tag}: {resolved_a} vs {resolved_b}  •  ROPE = ±{rope_percent}%  •  N = {n_blocks}  •  "
            f"p({resolved_a}) = {p_left_val:.3f}, p(rope) = {p_rope_val:.3f}, p({resolved_b}) = {p_right_val:.3f}"
        )

        return BayesianRankResult(
            metric=metric,
            scope=scope,
            scope_label=scope_label,
            bayesian_method=normalized_method,
            metrics_list=metrics_list,
            instances_list=instances_list,
            metrics_label=metrics_label,
            instances_label=instances_label,
            params_label=params_label,
            k=k,
            n_blocks=n_blocks,
            rope_percent=rope_percent,
            reference_algo=reference_algo,
            maximize=maximize,
            algo_a=resolved_a,
            algo_b=resolved_b,
            algorithms=algorithms,
            p_left=p_left_val,
            p_rope=p_rope_val,
            p_right=p_right_val,
            label_left=f"p({resolved_a}) = {p_left_val:.3f}",
            label_rope=f"p(rope) = {p_rope_val:.3f}",
            label_right=f"p({resolved_b}) = {p_right_val:.3f}",
            points_x=[round(float(v), 5) for v in x_sorted.tolist()],
            points_y=[round(float(v), 5) for v in y_sorted.tolist()],
            points_density=[round(float(v), 4) for v in d_sorted.tolist()],
            points_p_left=[round(float(v), 4) for v in p_left_sorted.tolist()],
            points_p_rope=[round(float(v), 4) for v in p_rope_sorted.tolist()],
            points_p_right=[round(float(v), 4) for v in p_right_sorted.tolist()],
            triangle_x=triangle_x,
            triangle_y=triangle_y,
            boundary_lines=boundary_lines,
            items=items,
        )
