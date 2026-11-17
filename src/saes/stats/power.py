"""Experimental design, statistical power analysis, and sample size planning for stochastic algorithms."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    Dataset,
    PostHocPowerItem,
    PowerAnalysisResult,
    PowerCurveSeries,
    SampleSizeRecommendation,
)
from saes.stats.nonparametric import compute_vargha_delaney_a12


def calculate_statistical_power(
    n: int,
    effect_size: float,
    alpha: float = 0.05,
    paired: bool = False,
) -> float:
    """Calculate exact two-tailed statistical power (1 - beta) for a Student t-test comparison.

    Uses the exact non-central Student t distribution (scipy.stats.nct).

    Args:
        n: Number of repetitions (runs) per sample group (n >= 2).
        effect_size: Standardized mean difference (Cohen's d >= 0).
        alpha: Significance level (default 0.05).
        paired: Whether the observations are paired across runs.

    Returns:
        Statistical power in [alpha, 1.0].
    """
    if n < 2 or effect_size <= 1e-9:
        return float(alpha)

    abs_d = abs(effect_size)

    if paired:
        df = n - 1
        delta = abs_d * math.sqrt(n)
    else:
        df = 2 * n - 2
        delta = abs_d * math.sqrt(n / 2.0)

    if df <= 0:
        return float(alpha)

    try:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        # Cumulative probability in rejection regions
        power = float(1.0 - stats.nct.cdf(t_crit, df, delta) + stats.nct.cdf(-t_crit, df, delta))
        return max(float(alpha), min(1.0, power))
    except Exception:
        # Fallback to normal approximation if non-central t numerical overflow occurs
        se = math.sqrt(2.0 / n) if not paired else math.sqrt(1.0 / n)
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_power = (abs_d / se) - z_crit
        return max(float(alpha), min(1.0, float(stats.norm.cdf(z_power))))


def estimate_required_sample_size(
    target_power: float = 0.80,
    effect_size: float = 0.50,
    alpha: float = 0.05,
) -> SampleSizeRecommendation:
    """Estimate required sample size (runs per instance) for independent and paired tests.

    Applies the Hodges-Lehmann asymptotic relative efficiency (A.R.E. = 0.864) bound
    for non-parametric rank-sum comparisons.

    Args:
        target_power: Target power 1 - beta (default 0.80).
        effect_size: Target standardized effect size Cohen's d.
        alpha: Significance level (default 0.05).

    Returns:
        SampleSizeRecommendation object.
    """
    abs_d = max(0.05, abs(effect_size))

    # Label effect size magnitude according to Cohen (1988)
    if abs_d < 0.35:
        label = "Small (d ≈ 0.20)"
    elif abs_d < 0.65:
        label = "Medium (d ≈ 0.50)"
    else:
        label = "Large (d ≈ 0.80)"

    # Binary search / linear scan for independent test
    n_ind = 2
    while n_ind < 2000:
        p = calculate_statistical_power(n_ind, abs_d, alpha=alpha, paired=False)
        if p >= target_power:
            break
        n_ind += 1

    # Linear scan for paired test
    n_paired = 2
    while n_paired < 2000:
        p = calculate_statistical_power(n_paired, abs_d, alpha=alpha, paired=True)
        if p >= target_power:
            break
        n_paired += 1

    # Non-parametric adjustment: Hodges-Lehmann lower bound (0.864) for continuous distributions
    n_nonparam = max(n_ind, math.ceil(n_ind / 0.864))

    return SampleSizeRecommendation(
        target_power=target_power,
        alpha=alpha,
        effect_size=round(abs_d, 3),
        effect_size_label=label,
        recommended_n_independent=n_ind,
        recommended_n_paired=n_paired,
        recommended_n_nonparametric=n_nonparam,
    )


def generate_power_curves(
    alpha: float = 0.05,
    custom_effect_size: float | None = None,
) -> list[PowerCurveSeries]:
    """Generate sample size vs power curves for standard and custom effect sizes."""
    sample_sizes = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]

    curves_specs = [
        ("Small Effect (d = 0.20)", 0.20),
        ("Medium Effect (d = 0.50)", 0.50),
        ("Large Effect (d = 0.80)", 0.80),
    ]

    if custom_effect_size is not None and custom_effect_size not in (0.2, 0.5, 0.8):
        curves_specs.append((f"Custom Effect (d = {custom_effect_size:.2f})", custom_effect_size))

    series_list: list[PowerCurveSeries] = []
    for label, d_val in curves_specs:
        powers = [
            round(calculate_statistical_power(n, d_val, alpha=alpha, paired=False), 4)
            for n in sample_sizes
        ]
        series_list.append(
            PowerCurveSeries(
                effect_size_label=label,
                effect_size_value=d_val,
                sample_sizes=sample_sizes,
                power_values=powers,
            )
        )

    return series_list


def compute_post_hoc_power_audit(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
) -> tuple[list[PostHocPowerItem], str]:
    """Audit achieved statistical power for all pairwise algorithm comparisons in the dataset."""
    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]
    metric_df = dataset.df[dataset.df["metric"] == target_metric].copy()
    maximize: bool = bool(metric_df["maximize"].iloc[0]) if not metric_df.empty else True

    algorithms = sorted(metric_df["algorithm"].unique().tolist())
    instances = sorted(metric_df["instance"].unique().tolist())

    audit_items: list[PostHocPowerItem] = []
    optimal_count = 0
    total_comparisons = 0

    for inst in instances:
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                a1 = algorithms[i]
                a2 = algorithms[j]

                s1 = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == a1)]["value"].to_numpy(
                    dtype=float
                )
                s2 = metric_df[(metric_df["instance"] == inst) & (metric_df["algorithm"] == a2)]["value"].to_numpy(
                    dtype=float
                )

                n1, n2 = len(s1), len(s2)
                if n1 < 2 or n2 < 2:
                    continue

                total_comparisons += 1
                n_eff = min(n1, n2)

                # Compute pooled standard deviation and Cohen's d
                m1, m2 = float(np.mean(s1)), float(np.mean(s2))
                var1, var2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))
                pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, n1 + n2 - 2))

                diff = abs(m1 - m2)
                cohens_d = (diff / pooled_sd) if pooled_sd > 1e-9 else 0.0

                # Vargha-Delaney A12
                a12_val, _ = compute_vargha_delaney_a12(s1, s2, maximize=maximize)

                # Achieved power
                power = calculate_statistical_power(n_eff, cohens_d, alpha=alpha, paired=(n1 == n2))

                # Check p-value
                try:
                    res = stats.ttest_ind(s1, s2, equal_var=False)
                    p_val = float(res.pvalue)
                except Exception:
                    p_val = 1.0

                is_sig = p_val < alpha

                if power >= 0.80:
                    status = "Optimal"
                    optimal_count += 1
                    recommendation = f"Statistical power is robust ({power:.1%}). Sample size N={n_eff} is reliable."
                elif power >= 0.70:
                    status = "Adequate"
                    recommendation = (
                        f"Moderate power ({power:.1%}). Consider increasing repetitions to N=30+ for safety."
                    )
                else:
                    status = "Underpowered"
                    rec_spec = estimate_required_sample_size(
                        target_power=0.80, effect_size=max(0.2, cohens_d), alpha=alpha
                    )
                    if not is_sig:
                        recommendation = (
                            f"High Type II error risk (β={1.0 - power:.1%}). Small sample size N={n_eff}. "
                            f"Recommended sample size: N={rec_spec.recommended_n_independent} runs."
                        )
                    else:
                        recommendation = (
                            f"Difference detected, but power is below 80% ({power:.1%}). "
                            f"Confirm with N={rec_spec.recommended_n_independent} runs."
                        )

                audit_items.append(
                    PostHocPowerItem(
                        algo_1=a1,
                        algo_2=a2,
                        instance=inst,
                        metric=target_metric,
                        n_samples=n_eff,
                        cohens_d=round(cohens_d, 4),
                        a12=round(a12_val, 4),
                        achieved_power=round(power, 4),
                        power_status=status,
                        is_significant=is_sig,
                        recommendation=recommendation,
                    )
                )

    if total_comparisons == 0:
        overall_health = "Optimal"
    else:
        ratio = optimal_count / total_comparisons
        if ratio >= 0.75:
            overall_health = "Optimal"
        elif ratio >= 0.50:
            overall_health = "Adequate"
        else:
            overall_health = "Underpowered"

    return audit_items, overall_health


def analyze_statistical_power(
    dataset: Dataset,
    metric: str | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    target_effect_size: float = 0.50,
) -> PowerAnalysisResult:
    """Run full statistical power analysis, a priori recommendations, and post-hoc audit.

    Args:
        dataset: Validated SAES Dataset.
        metric: Evaluation metric identifier (default first metric).
        alpha: Significance level (default 0.05).
        target_power: Desired statistical power (default 0.80).
        target_effect_size: Expected Cohen's d (default 0.50 medium effect).

    Returns:
        PowerAnalysisResult object.
    """
    if not dataset.metrics:
        raise StatisticalAnalysisError("Dataset contains no metrics for power analysis.")

    target_metric = metric if metric in dataset.metrics else dataset.metrics[0]

    # 1. A priori recommendations for standard benchmarks
    standard_effects = [
        (0.20, "Small (d = 0.20)"),
        (0.50, "Medium (d = 0.50)"),
        (0.80, "Large (d = 0.80)"),
    ]
    if target_effect_size not in (0.20, 0.50, 0.80):
        standard_effects.append((target_effect_size, f"Custom (d = {target_effect_size:.2f})"))

    recommendations: list[SampleSizeRecommendation] = [
        estimate_required_sample_size(target_power=target_power, effect_size=d_val, alpha=alpha)
        for d_val, _ in standard_effects
    ]

    # 2. Power curve series
    power_curves = generate_power_curves(alpha=alpha, custom_effect_size=target_effect_size)

    # 3. Post-hoc audit
    audit_items, overall_health = compute_post_hoc_power_audit(
        dataset=dataset,
        metric=target_metric,
        alpha=alpha,
    )

    return PowerAnalysisResult(
        alpha=alpha,
        target_power=target_power,
        target_effect_size=target_effect_size,
        sample_size_recommendations=recommendations,
        power_curves=power_curves,
        post_hoc_audit=audit_items,
        overall_power_health=overall_health,
    )
