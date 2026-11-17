"""Non-parametric hypothesis tests, effect sizes (Vargha-Delaney A12, Cliff's delta), and p-value corrections."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import EffectSize, StatisticalResult
from saes.stats.assumptions import HomogeneityChecker, NormalityChecker
from saes.stats.base import StatisticalTest
from saes.stats.ttest import _compute_group_summary


def classify_vargha_delaney_a12(a12: float) -> str:
    """Classify Vargha-Delaney A12 effect size magnitude (Vargha & Delaney 2000; Arcuri & Briand 2014).

    Thresholds for |A12 - 0.5|:
      - < 0.06 (0.44 < A12 < 0.56): negligible
      - < 0.14 (0.36 < A12 < 0.64): small
      - < 0.21 (0.29 < A12 < 0.71): medium
      - >= 0.21 (A12 <= 0.29 or A12 >= 0.71): large
    """
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    if diff < 0.14:
        return "small"
    if diff < 0.21:
        return "medium"
    return "large"


def compute_vargha_delaney_a12(
    x1: np.ndarray,
    x2: np.ndarray,
    maximize: bool = True,
) -> tuple[float, str]:
    """Compute Vargha-Delaney A12 non-parametric stochastic superiority effect size.

    A12 measures the probability that a randomly selected observation from group 1
    is better than a randomly selected observation from group 2 (plus 0.5 * tie probability).
    """
    c1 = x1[~np.isnan(x1)]
    c2 = x2[~np.isnan(x2)]
    n1, n2 = len(c1), len(c2)
    if n1 == 0 or n2 == 0:
        return 0.5, "negligible"

    # Pairwise comparison matrix: shape (n1, n2)
    diff_matrix = c1[:, None] - c2[None, :]
    if not maximize:
        diff_matrix = -diff_matrix

    better = float(np.sum(diff_matrix > 0))
    ties = float(np.sum(diff_matrix == 0))
    a12 = float((better + 0.5 * ties) / float(n1 * n2))
    return a12, classify_vargha_delaney_a12(a12)


def compute_cliffs_delta(
    x1: np.ndarray,
    x2: np.ndarray,
    maximize: bool = True,
) -> tuple[float, str]:
    """Compute Cliff's delta = 2 * A12 - 1 in [-1, 1]."""
    a12, mag = compute_vargha_delaney_a12(x1, x2, maximize=maximize)
    delta = float(2.0 * a12 - 1.0)
    return delta, mag


def _classify_epsilon_squared(eps_sq: float) -> str:
    """Classify Kruskal-Wallis epsilon-squared effect size."""
    if eps_sq < 0.01:
        return "negligible"
    if eps_sq < 0.08:
        return "small"
    if eps_sq < 0.26:
        return "medium"
    return "large"


def adjust_pvalues(p_values: list[float], method: str = "holm") -> list[float]:
    """Apply multiple-testing p-value correction across a family of hypotheses.

    Supported methods:
        - 'holm': Holm-Bonferroni step-down procedure (Demšar 2006)
        - 'bonferroni': Bonferroni-Dunn correction
        - 'hochberg': Hochberg step-up procedure (García & Herrera 2008)
        - 'fdr_bh': Benjamini-Hochberg False Discovery Rate
        - 'none': Unadjusted p-values
    """
    m = len(p_values)
    if m == 0:
        return []
    clean_p = [min(1.0, max(0.0, float(p) if math.isfinite(p) else 1.0)) for p in p_values]
    if m == 1 or method.lower() in {"none", "unadjusted"}:
        return clean_p

    norm_method = method.strip().lower()
    arr = np.array(clean_p, dtype=float)
    order = np.argsort(arr)
    sorted_p = arr[order]
    adjusted = np.zeros(m, dtype=float)

    if norm_method == "bonferroni":
        adjusted = np.minimum(1.0, arr * float(m))
        return [float(x) for x in adjusted.tolist()]

    if norm_method in {"holm", "holm-bonferroni"}:
        # Step-down: p_adj[i] = max_{j <= i} min(1, (m - j) * p_(j))
        running_max = 0.0
        for idx in range(m):
            val = min(1.0, sorted_p[idx] * float(m - idx))
            running_max = max(running_max, val)
            adjusted[order[idx]] = running_max
        return [float(x) for x in adjusted.tolist()]

    if norm_method == "hochberg":
        # Step-up: from largest to smallest p-value
        running_min = 1.0
        for idx in range(m - 1, -1, -1):
            val = min(1.0, sorted_p[idx] * float(m - idx))
            running_min = min(running_min, val)
            adjusted[order[idx]] = running_min
        return [float(x) for x in adjusted.tolist()]

    if norm_method in {"fdr_bh", "bh", "benjamini-hochberg"}:
        running_min = 1.0
        for idx in range(m - 1, -1, -1):
            rank = float(idx + 1)
            val = min(1.0, sorted_p[idx] * float(m) / rank)
            running_min = min(running_min, val)
            adjusted[order[idx]] = running_min
        return [float(x) for x in adjusted.tolist()]

    # Default fallback to Holm
    return adjust_pvalues(clean_p, method="holm")


class WilcoxonSignedRankTest(StatisticalTest):
    """Wilcoxon Signed-Rank non-parametric paired two-sample test."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"WilcoxonSignedRankTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        if len(x1) != len(x2):
            raise StatisticalAnalysisError(
                f"WilcoxonSignedRankTest requires equal paired sample sizes. Got N1={len(x1)}, N2={len(x2)}."
            )

        n = len(x1)
        if n < 2:
            raise StatisticalAnalysisError("WilcoxonSignedRankTest requires at least 2 paired observations.")

        diffs = x1 - x2
        normality_res = NormalityChecker(alpha=alpha).check({"Paired Differences": diffs})
        assumptions = tuple(normality_res)

        warnings_list: list[str] = []
        if n < 6:
            warnings_list.append(
                f"Small paired sample size (N={n}). Exact Wilcoxon Signed-Rank critical values have limited power."
            )

        if np.all(diffs == 0):
            w_stat = 0.0
            p_val = 1.0
        else:
            try:
                res = stats.wilcoxon(x1, x2, zero_method="zsplit", alternative="two-sided")
                w_stat = float(res.statistic)
                p_val = float(res.pvalue)
            except Exception:
                w_stat = 0.0
                p_val = 1.0

        a12, a12_mag = compute_vargha_delaney_a12(x1, x2, maximize=True)
        effect_size = EffectSize(
            name="Vargha-Delaney A12",
            value=a12,
            magnitude=a12_mag,
        )

        is_sig = p_val < alpha
        med1, med2 = float(np.median(x1)), float(np.median(x2))
        interp = (
            f"Statistically significant paired non-parametric difference detected between '{g1_name}' "
            f"(median={med1:.4g}) and '{g2_name}' (median={med2:.4g}) via Wilcoxon Signed-Rank test "
            f"(W = {w_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {a12:.4f} ({a12_mag})."
            if is_sig
            else f"No statistically significant paired non-parametric difference detected between '{g1_name}' "
            f"and '{g2_name}' via Wilcoxon Signed-Rank test (W = {w_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=w_stat,
            p_value=p_val,
            degrees_of_freedom=float(n),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=tuple(warnings_list),
            interpretation=interp,
            group_summaries=summaries,
        )


class MannWhitneyUTest(StatisticalTest):
    """Mann-Whitney U (Wilcoxon Rank-Sum) non-parametric two-sample independent test."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) != 2:
            raise StatisticalAnalysisError(f"MannWhitneyUTest requires exactly 2 groups, got {len(groups)}.")

        keys = list(groups.keys())
        g1_name, g2_name = keys[0], keys[1]
        x1 = groups[g1_name][~np.isnan(groups[g1_name])]
        x2 = groups[g2_name][~np.isnan(groups[g2_name])]

        n1, n2 = len(x1), len(x2)
        if n1 < 2 or n2 < 2:
            raise StatisticalAnalysisError("Each group must contain at least 2 observations for MannWhitneyUTest.")

        normality_res = NormalityChecker(alpha=alpha).check(groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        res = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)

        a12, a12_mag = compute_vargha_delaney_a12(x1, x2, maximize=True)
        effect_size = EffectSize(
            name="Vargha-Delaney A12",
            value=a12,
            magnitude=a12_mag,
        )

        is_sig = p_val < alpha
        med1, med2 = float(np.median(x1)), float(np.median(x2))
        interp = (
            f"Statistically significant non-parametric difference detected between '{g1_name}' "
            f"(median={med1:.4g}) and '{g2_name}' (median={med2:.4g}) via Mann-Whitney U test "
            f"(U = {u_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {a12:.4f} ({a12_mag})."
            if is_sig
            else f"No statistically significant non-parametric difference detected between '{g1_name}' "
            f"and '{g2_name}' via Mann-Whitney U test (U = {u_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {g1_name: _compute_group_summary(x1), g2_name: _compute_group_summary(x2)}

        return StatisticalResult(
            test_name="Mann-Whitney U Test (Wilcoxon Rank-Sum)",
            statistic=u_stat,
            p_value=p_val,
            degrees_of_freedom=(float(n1), float(n2)),
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=(),
            interpretation=interp,
            group_summaries=summaries,
        )


class KruskalWallisTest(StatisticalTest):
    """Kruskal-Wallis H non-parametric omnibus test across 2 or more independent groups."""

    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        if len(groups) < 2:
            raise StatisticalAnalysisError(f"KruskalWallisTest requires at least 2 groups, got {len(groups)}.")

        clean_groups = {k: v[~np.isnan(v)] for k, v in groups.items()}
        for grp_name, v in clean_groups.items():
            if len(v) < 2:
                raise StatisticalAnalysisError(
                    f"Group '{grp_name}' has only {len(v)} observations (minimum 2 required)."
                )

        normality_res = NormalityChecker(alpha=alpha).check(clean_groups)
        homogeneity_res = HomogeneityChecker(alpha=alpha).check(clean_groups)
        assumptions = tuple(normality_res + [homogeneity_res])

        group_list = list(clean_groups.values())
        k = len(group_list)
        n_total = sum(len(g) for g in group_list)
        df_between = float(k - 1)

        all_vals = np.concatenate(group_list)
        if np.all(all_vals == all_vals[0]):
            h_stat = 0.0
            p_val = 1.0
            eps_sq = 0.0
        else:
            try:
                res = stats.kruskal(*group_list)
                h_stat = float(res.statistic)
                p_val = float(res.pvalue)
                eps_sq = float(h_stat / float(n_total - 1)) if n_total > 1 else 0.0
            except Exception:
                h_stat = 0.0
                p_val = 1.0
                eps_sq = 0.0

        effect_size = EffectSize(
            name="Epsilon-squared (ε²)",
            value=eps_sq,
            magnitude=_classify_epsilon_squared(eps_sq),
        )

        is_sig = p_val < alpha
        interp = (
            f"Statistically significant non-parametric difference detected among {k} groups via Kruskal-Wallis H test "
            f"(H({df_between:.0f}) = {h_stat:.4f}, p = {p_val:.4e}, alpha = {alpha}). "
            f"Effect size: {effect_size.name} = {eps_sq:.4f} ({effect_size.magnitude})."
            if is_sig
            else f"No statistically significant difference detected among {k} groups via Kruskal-Wallis H test "
            f"(H({df_between:.0f}) = {h_stat:.4f}, p = {p_val:.4f}, alpha = {alpha})."
        )

        summaries = {name: _compute_group_summary(data) for name, data in clean_groups.items()}

        return StatisticalResult(
            test_name="Kruskal-Wallis H Test",
            statistic=h_stat,
            p_value=p_val,
            degrees_of_freedom=df_between,
            alpha=alpha,
            is_significant=is_sig,
            effect_size=effect_size,
            confidence_interval=None,
            assumptions=assumptions,
            warnings=(),
            interpretation=interp,
            group_summaries=summaries,
        )
