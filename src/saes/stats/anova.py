"""One-Way ANOVA implementations for SAES (Fisher One-Way ANOVA, Welch's ANOVA)."""

import numpy as np
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import (
    EffectSize,
    StatisticalResult,
)
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.base import StatisticalTest
from saes.stats.ttest import _compute_group_summary


def _classify_eta_squared(eta_sq: float) -> str:
    """Classify magnitude of Eta-squared effect size."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


class OneWayANOVA(StatisticalTest):
    """Standard Fisher One-Way ANOVA (assumes equal variances and normality)."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) < 2:
            raise StatisticalAnalysisError(f"OneWayANOVA requires at least 2 groups, got {len(groups)}.")

        clean_groups = {k: v[~np.isnan(v)] for k, v in groups.items()}
        for grp_name, v in clean_groups.items():
            if len(v) < 2:
                msg = f"Group '{grp_name}' has only {len(v)} observations (minimum 2 required)."
                raise StatisticalAnalysisError(msg)

        normality_res = NormalityChecker(alpha=alpha).check(clean_groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(clean_groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        warnings_list: list[str] = []
        if not homogeneity_res.passed:
            warnings_list.append("Homogeneity of variance assumption violated. Welch's ANOVA is recommended.")
        if any(not n.passed for n in normality_res):
            warnings_list.append("Normality assumption violated in one or more groups.")

        group_data_list = list(clean_groups.values())
        k = len(group_data_list)
        n_total = sum(len(g) for g in group_data_list)
        df_between = float(k - 1)
        df_within = float(n_total - k)

        res = stats.f_oneway(*group_data_list)
        f_stat = float(res.statistic)
        p_val = float(res.pvalue)

        # Sum of Squares calculation for Eta-squared & Omega-squared
        all_obs = np.concatenate(group_data_list)
        grand_mean = np.mean(all_obs)

        ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data_list))
        ss_total = float(sum((x - grand_mean) ** 2 for x in all_obs))

        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
        ms_within = float((ss_total - ss_between) / df_within) if df_within > 0 else 0.0
        denom = ss_total + ms_within
        omega_sq = float((ss_between - (k - 1) * ms_within) / denom) if denom > 0 else 0.0

        effect_size = EffectSize(
            name="Eta-squared (η²)",
            value=eta_sq,
            magnitude=_classify_eta_squared(eta_sq),
        )

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant difference detected among {k} groups via One-Way ANOVA "
            f"(F({df_between:.0f}, {df_within:.0f}) = {f_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {eta_sq:.4f} ({effect_size.magnitude}), Omega-squared = {omega_sq:.4f}."
            if is_sig
            else f"No statistically significant difference detected among {k} groups via One-Way ANOVA "
            f"(F({df_between:.0f}, {df_within:.0f}) = {f_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {name: _compute_group_summary(data) for name, data in clean_groups.items()}

        return StatisticalResult(
            test_name="One-Way Analysis of Variance (ANOVA)",
            statistic=f_stat,
            p_value=p_val,
            degrees_of_freedom=(df_between, df_within),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )


class WelchANOVA(StatisticalTest):
    """Welch's One-Way ANOVA (handles heteroscedasticity across groups)."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) < 2:
            raise StatisticalAnalysisError(f"WelchANOVA requires at least 2 groups, got {len(groups)}.")

        clean_groups = {k: v[~np.isnan(v)] for k, v in groups.items()}
        for grp_name, v in clean_groups.items():
            if len(v) < 2:
                msg = f"Group '{grp_name}' has only {len(v)} observations (minimum 2 required)."
                raise StatisticalAnalysisError(msg)

        normality_res = NormalityChecker(alpha=alpha).check(clean_groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(clean_groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        warnings_list: list[str] = []
        if any(not n.passed for n in normality_res):
            warnings_list.append("Normality assumption violated in one or more groups.")

        group_list = list(clean_groups.values())
        k = len(group_list)
        ni = np.array([len(g) for g in group_list], dtype=float)
        means = np.array([np.mean(g) for g in group_list], dtype=float)
        vars_ = np.array([np.var(g, ddof=1) for g in group_list], dtype=float)
        if np.any(vars_ == 0):
            warnings_list.append(
                "One or more groups have zero variance (constant values). Standard One-Way ANOVA was used."
            )
            res = OneWayANOVA().run(clean_groups, alpha=alpha)
            all_warnings = tuple(list(res.warnings) + warnings_list)
            return StatisticalResult(
                test_name="Welch's One-Way Analysis of Variance (ANOVA)",
                statistic=res.statistic,
                p_value=res.p_value,
                degrees_of_freedom=res.degrees_of_freedom,
                alpha=alpha,
                is_significant=res.is_significant,
                effect_size=res.effect_size,
                confidence_interval=res.confidence_interval,
                assumptions=assumptions,
                warnings=all_warnings,
                interpretation=res.interpretation,
                group_summaries=res.group_summaries,
            )

        weights = ni / vars_
        sum_weights = np.sum(weights)
        weighted_mean = np.sum(weights * means) / sum_weights

        df_between = float(k - 1)
        term1 = np.sum((1.0 - weights / sum_weights) ** 2 / (ni - 1.0))
        lambda_val = (3.0 / (k**2 - 1.0)) * term1
        df_within = float(1.0 / lambda_val) if lambda_val > 0 else 1.0

        f_stat = float(
            (1.0 / (k - 1.0))
            * np.sum(weights * (means - weighted_mean) ** 2)
            / (1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * term1)
        )
        p_val = float(1.0 - stats.f.cdf(f_stat, df_between, df_within))

        all_obs = np.concatenate(group_list)
        grand_mean = np.mean(all_obs)
        ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_list))
        ss_total = float(sum((x - grand_mean) ** 2 for x in all_obs))
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        effect_size = EffectSize(
            name="Eta-squared (η²)",
            value=eta_sq,
            magnitude=_classify_eta_squared(eta_sq),
        )

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant difference detected among {k} groups via Welch's ANOVA "
            f"(F({df_between:.0f}, {df_within:.2f}) = {f_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {eta_sq:.4f} ({effect_size.magnitude})."
            if is_sig
            else f"No statistically significant difference detected among {k} groups via Welch's ANOVA "
            f"(F({df_between:.0f}, {df_within:.2f}) = {f_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {name: _compute_group_summary(data) for name, data in clean_groups.items()}

        return StatisticalResult(
            test_name="Welch's One-Way Analysis of Variance (ANOVA)",
            statistic=f_stat,
            p_value=p_val,
            degrees_of_freedom=(df_between, df_within),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )
