"""Dolan-Moré performance profiles, data profiles, and global meta-benchmarking."""

from __future__ import annotations

import math

import numpy as np

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    AlgorithmProfileCurve,
    Dataset,
    GlobalMetaRankingItem,
    PerformanceProfilesResult,
    WinTieLossCell,
)
from saes.stats.nonparametric import MannWhitneyUTest, WilcoxonSignedRankTest


def compute_performance_profiles(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    max_tau: float | None = None,
    num_tau_points: int = 60,
) -> PerformanceProfilesResult:
    """Compute Dolan-Moré performance profiles and global Win/Tie/Loss meta-ranking.

    Reference:
        Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software
        with performance profiles. Mathematical Programming, 91(2), 201-213.

    Args:
        dataset: Validated SAES Dataset.
        metric: Evaluation metric identifier (defaults to first metric).
        alpha: Statistical significance threshold for Win/Tie/Loss comparisons.
        max_tau: Optional upper bound for performance ratio tau.
        num_tau_points: Number of evaluation points along the tau axis.

    Returns:
        PerformanceProfilesResult with curves, W/T/L matrix, and meta-rankings.
    """
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no evaluation metrics.")

    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]
    metric_df = dataset.df[dataset.df["metric"] == target_metric].copy()

    if metric_df.empty:
        raise StatisticalAnalysisError(f"No records found for metric '{target_metric}'.")

    maximize: bool = bool(metric_df["maximize"].iloc[0])
    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    instances = sorted(metric_df["instance"].unique().tolist())

    if len(algorithms) < 1 or len(instances) < 1:
        raise StatisticalAnalysisError("Need at least 1 algorithm and 1 instance for performance profiling.")

    # 1. Compute representative performance per (instance, algorithm) cell
    # Use median (or mean) across runs
    perf_matrix: dict[str, dict[str, float]] = {}
    for inst in instances:
        perf_matrix[inst] = {}
        for algo in algorithms:
            sub = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == algo)]
            if not sub.empty:
                val = float(sub["value"].median())
            else:
                val = float("nan")
            perf_matrix[inst][algo] = val

    # 2. Compute performance ratios r_{p, s} per Dolan-Moré (2002)
    # Minimization: r_{p,s} = t_{p,s} / min_a t_{p,a}
    # Maximization: r_{p,s} = max_a t_{p,a} / t_{p,s}
    ratio_matrix: dict[str, dict[str, float]] = {inst: {} for inst in instances}
    all_finite_ratios: list[float] = []

    for inst in instances:
        row_vals = [perf_matrix[inst][a] for a in algorithms if not math.isnan(perf_matrix[inst][a])]
        if not row_vals:
            continue

        min_val = min(row_vals)
        max_val = max(row_vals)

        # Apply positive shift if any value is <= 0 to keep performance ratio well-defined
        shift = 0.0
        if min_val <= 0.0:
            shift = abs(min_val) + 1.0

        for algo in algorithms:
            val = perf_matrix[inst][algo]
            if math.isnan(val):
                ratio = float("inf")
            else:
                shifted_val = val + shift
                if maximize:
                    best_shifted = max_val + shift
                    ratio = best_shifted / max(1e-12, shifted_val)
                else:
                    best_shifted = min_val + shift
                    ratio = shifted_val / max(1e-12, best_shifted)

            # Round ratio to avoid floating precision artifacts near 1.0
            if abs(ratio - 1.0) < 1e-9:
                ratio = 1.0

            ratio_matrix[inst][algo] = ratio
            if math.isfinite(ratio):
                all_finite_ratios.append(ratio)

    if not all_finite_ratios:
        all_finite_ratios = [1.0]

    computed_max_tau = max(all_finite_ratios) if all_finite_ratios else 1.0
    upper_tau = max_tau if max_tau is not None else min(10.0, max(2.0, computed_max_tau))
    if upper_tau <= 1.0:
        upper_tau = 2.0

    # Create geometrically spaced or linear tau grid starting at 1.0
    tau_grid = np.unique(
        np.concatenate(
            [
                np.array([1.0]),
                np.linspace(1.0, upper_tau, num_tau_points),
                np.array(sorted(all_finite_ratios)),
            ]
        )
    )
    tau_grid = tau_grid[(tau_grid >= 1.0) & (tau_grid <= upper_tau)]
    tau_values = [round(float(t), 4) for t in sorted(set(tau_grid.tolist()))]

    # 3. Compute profile curves rho_s(tau) = fraction of problems where r_{p,s} <= tau
    total_instances = len(instances)
    curves: list[AlgorithmProfileCurve] = []

    for algo in algorithms:
        algo_ratios = [ratio_matrix[inst].get(algo, float("inf")) for inst in instances]
        rho_vals: list[float] = []
        for t in tau_values:
            solved = sum(1 for r in algo_ratios if r <= t)
            rho_vals.append(round(solved / total_instances, 4))

        efficiency = rho_vals[0] if rho_vals else 0.0  # rho(1.0)
        robustness = rho_vals[-1] if rho_vals else 0.0  # rho(tau_max)

        curves.append(
            AlgorithmProfileCurve(
                algorithm=algo,
                tau_values=tau_values,
                rho_values=rho_vals,
                efficiency=efficiency,
                robustness=robustness,
            )
        )

    # 4. Global Win / Tie / Loss (W/T/L) Score Matrix
    wtl_matrix: list[list[WinTieLossCell]] = []
    algo_wins: dict[str, int] = {a: 0 for a in algorithms}
    algo_ties: dict[str, int] = {a: 0 for a in algorithms}
    algo_losses: dict[str, int] = {a: 0 for a in algorithms}

    wilcoxon = WilcoxonSignedRankTest()
    mannwhitney = MannWhitneyUTest()

    for algo_row in algorithms:
        matrix_row: list[WinTieLossCell] = []
        for algo_col in algorithms:
            if algo_row == algo_col:
                matrix_row.append(
                    WinTieLossCell(
                        algo_row=algo_row,
                        algo_col=algo_col,
                        wins=0,
                        ties=total_instances,
                        losses=0,
                        net=0,
                    )
                )
                continue

            wins = 0
            ties = 0
            losses = 0

            for inst in instances:
                sub_r = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == algo_row)]
                sub_c = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == algo_col)]

                v_r = sub_r["value"].to_numpy(dtype=float)
                v_c = sub_c["value"].to_numpy(dtype=float)

                if len(v_r) == 0 or len(v_c) == 0:
                    ties += 1
                    continue

                # Run statistical hypothesis test
                try:
                    if len(v_r) == len(v_c) and len(v_r) >= 3:
                        res = wilcoxon.run({algo_row: v_r, algo_col: v_c}, alpha=alpha)
                    else:
                        res = mannwhitney.run({algo_row: v_r, algo_col: v_c}, alpha=alpha)
                    p_val = res.p_value
                except Exception:
                    p_val = 1.0

                if p_val < alpha:
                    mean_r = float(np.mean(v_r))
                    mean_c = float(np.mean(v_c))
                    if maximize:
                        if mean_r > mean_c:
                            wins += 1
                        else:
                            losses += 1
                    else:
                        if mean_r < mean_c:
                            wins += 1
                        else:
                            losses += 1
                else:
                    ties += 1

            net = wins - losses
            algo_wins[algo_row] += wins
            algo_ties[algo_row] += ties
            algo_losses[algo_row] += losses

            matrix_row.append(
                WinTieLossCell(
                    algo_row=algo_row,
                    algo_col=algo_col,
                    wins=wins,
                    ties=ties,
                    losses=losses,
                    net=net,
                )
            )
        wtl_matrix.append(matrix_row)

    # 5. Global Meta-Rankings
    curve_map = {c.algorithm: c for c in curves}
    num_comparisons = max(1, (len(algorithms) - 1) * total_instances)

    meta_items: list[GlobalMetaRankingItem] = []
    for algo in algorithms:
        c = curve_map.get(algo)
        eff = c.efficiency if c else 0.0
        rob = c.robustness if c else 0.0
        win_cnt = algo_wins[algo]
        tie_cnt = algo_ties[algo]
        loss_cnt = algo_losses[algo]
        net = win_cnt - loss_cnt
        copeland = round(net / num_comparisons, 4)

        meta_items.append(
            GlobalMetaRankingItem(
                rank=1,
                algorithm=algo,
                efficiency=eff,
                robustness=rob,
                total_wins=win_cnt,
                total_ties=tie_cnt,
                total_losses=loss_cnt,
                net_score=net,
                copeland_score=copeland,
            )
        )

    # Sort meta-rankings: primarily by Net Score, then by Efficiency, then Robustness
    meta_items.sort(key=lambda x: (x.net_score, x.efficiency, x.robustness), reverse=True)
    for rank_idx, item in enumerate(meta_items, start=1):
        object.__setattr__(item, "rank", rank_idx)

    return PerformanceProfilesResult(
        metric=target_metric,
        maximize=maximize,
        alpha=alpha,
        tau_values=tau_values,
        curves=curves,
        wtl_matrix=wtl_matrix,
        meta_rankings=meta_items,
    )
