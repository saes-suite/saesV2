"""Two-sample t-test implementations for SAES (Student's, Welch's, Paired)."""

import numpy as np
from scipy import stats

from saesV2.domain.exceptions import StatisticalAnalysisError
from saesV2.domain.models import (
    EffectSize,
    StatisticalResult,
)
from saesV2.stats.assumptions import HomogeneityChecker, NormalityChecker
from saesV2.stats.base import StatisticalTest


def _compute_group_summary(data: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for a 1D numeric array."""
    clean = data[~np.isnan(data)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan}
    q75, q25 = np.percentile(clean, [75, 25])
    return {
        "n": float(n),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(clean)),
        "iqr": float(q75 - q25),
    }


def _classify_cohen_d(d_val: float) -> str:
    """Classify magnitude of Cohen's d or Hedges' g."""
    abs_d = abs(d_val)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


class IndependentTTest(StatisticalTest):
    """Student's two-sample independent t-test (assumes equal variances)."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"IndependentTTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        n1, n2 = len(x1), len(x2)
        if n1 < 2 or n2 < 2:
            raise StatisticalAnalysisError("Each group must contain at least 2 observations for IndependentTTest.")

        # Assumptions
        normality_res = NormalityChecker(alpha=alpha).check(groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        warnings_list: list[str] = []
        if not homogeneity_res.passed:
            warnings_list.append("Homogeneity of variance assumption violated. Welch's t-test is recommended.")
        if any(not n.passed for n in normality_res):
            warnings_list.append("Normality assumption violated in one or more groups.")

        # Calculate Student's t-test
        m1, m2 = np.mean(x1), np.mean(x2)
        v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        df = float(n1 + n2 - 2)

        pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se_diff = np.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))

        if se_diff == 0:
            t_stat = 0.0
            p_val = 1.0
            ci = (float(m1 - m2), float(m1 - m2))
            d_val = 0.0
        else:
            res = stats.ttest_ind(x1, x2, equal_var=True)
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)
            tcrit = stats.t.ppf(1 - alpha / 2, df)
            diff = m1 - m2
            ci = (float(diff - tcrit * se_diff), float(diff + tcrit * se_diff))
            d_val = float(diff / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0

        effect_size = EffectSize(
            name="Cohen's d",
            value=d_val,
            magnitude=_classify_cohen_d(d_val),
        )

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant difference detected between '{g1_name}' and '{g2_name}' "
            f"(t({df:.1f}) = {t_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {d_val:.4f} ({effect_size.magnitude})."
            if is_sig
            else f"No statistically significant difference detected between '{g1_name}' and '{g2_name}' "
            f"(t({df:.1f}) = {t_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Student's Independent Two-Sample t-test",
            statistic=t_stat,
            p_value=p_val,
            degrees_of_freedom=df,
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=ci,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )


class WelchTTest(StatisticalTest):
    """Welch's two-sample t-test (does not assume equal variances)."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"WelchTTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        n1, n2 = len(x1), len(x2)
        if n1 < 2 or n2 < 2:
            raise StatisticalAnalysisError("Each group must contain at least 2 observations for WelchTTest.")

        normality_res = NormalityChecker(alpha=alpha).check(groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        warnings_list: list[str] = []
        if any(not n.passed for n in normality_res):
            warnings_list.append("Normality assumption violated in one or more groups.")

        m1, m2 = np.mean(x1), np.mean(x2)
        v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

        # Welch-Satterthwaite degrees of freedom
        se1_sq = v1 / n1
        se2_sq = v2 / n2
        se_diff = np.sqrt(se1_sq + se2_sq)

        if se_diff == 0:
            t_stat = 0.0
            p_val = 1.0
            df = float(n1 + n2 - 2)
            ci = (float(m1 - m2), float(m1 - m2))
            hedges_g = 0.0
        else:
            res = stats.ttest_ind(x1, x2, equal_var=False)
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)

            df_num = (se1_sq + se2_sq) ** 2
            df_den = (se1_sq**2) / (n1 - 1) + (se2_sq**2) / (n2 - 1)
            df = float(df_num / df_den) if df_den > 0 else float(n1 + n2 - 2)

            tcrit = stats.t.ppf(1 - alpha / 2, df)
            diff = m1 - m2
            ci = (float(diff - tcrit * se_diff), float(diff + tcrit * se_diff))

            # Hedges' g (with small sample bias correction factor J)
            pooled_s = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            cohen_d = diff / pooled_s if pooled_s > 0 else 0.0
            j_factor = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2) > 2 else 1.0
            hedges_g = float(cohen_d * j_factor)

        effect_size = EffectSize(
            name="Hedges' g",
            value=hedges_g,
            magnitude=_classify_cohen_d(hedges_g),
        )

        is_sig = p_val < alpha
        if is_sig:
            interp = (
                f"Statistically significant difference detected between '{g1_name}' and '{g2_name}' via Welch's t-test "
                f"(t({df:.2f}) = {t_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
                f"Effect size: {effect_size.name} = {hedges_g:.4f} ({effect_size.magnitude})."
            )
        else:
            interp = (
                f"No statistically significant difference detected between '{g1_name}' and '{g2_name}' via Welch's "
                f"t-test (t({df:.2f}) = {t_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
            )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Welch's Two-Sample t-test (Unequal Variances)",
            statistic=t_stat,
            p_value=p_val,
            degrees_of_freedom=df,
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=ci,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )


class PairedTTest(StatisticalTest):
    """Paired samples t-test for dependent observations."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"PairedTTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        if len(x1) != len(x2):
            raise StatisticalAnalysisError(
                f"PairedTTest requires equal sample sizes across paired groups. Got N1={len(x1)}, N2={len(x2)}."
            )

        n = len(x1)
        if n < 2:
            raise StatisticalAnalysisError("PairedTTest requires at least 2 paired observations.")

        diffs = x1 - x2
        normality_res = NormalityChecker(alpha=alpha).check({"Paired Differences": diffs})
        assumptions = tuple(normality_res)

        warnings_list: list[str] = []
        if not normality_res[0].passed:
            warnings_list.append("Normality of paired differences assumption is violated.")

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        df = float(n - 1)

        if std_diff == 0:
            t_stat = 0.0
            p_val = 1.0
            ci = (float(mean_diff), float(mean_diff))
            cohen_d = 0.0
        else:
            res = stats.ttest_rel(x1, x2)
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)

            se_diff = std_diff / np.sqrt(n)
            tcrit = stats.t.ppf(1 - alpha / 2, df)
            ci = (float(mean_diff - tcrit * se_diff), float(mean_diff + tcrit * se_diff))
            cohen_d = float(mean_diff / std_diff)

        effect_size = EffectSize(
            name="Cohen's d (paired)",
            value=cohen_d,
            magnitude=_classify_cohen_d(cohen_d),
        )

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant paired difference detected between '{g1_name}' and '{g2_name}' "
            f"(t({df:.0f}) = {t_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {cohen_d:.4f} ({effect_size.magnitude})."
            if is_sig
            else f"No statistically significant paired difference detected between '{g1_name}' and '{g2_name}' "
            f"(t({df:.0f}) = {t_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Paired Samples t-test",
            statistic=t_stat,
            p_value=p_val,
            degrees_of_freedom=df,
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=ci,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )
