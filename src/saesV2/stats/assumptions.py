"""Statistical assumption testing engine for SAES."""

import numpy as np
from scipy import stats

from saesV2.domain.models import AssumptionResult


class NormalityChecker:
    """Evaluates normality assumption using Shapiro-Wilk test for each group."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def check(self, groups: dict[str, np.ndarray]) -> list[AssumptionResult]:
        """Check normality for all groups in dictionary.

        Args:
            groups: Dictionary mapping group names to 1D numpy arrays.

        Returns:
            List of AssumptionResult objects for each group.
        """
        results: list[AssumptionResult] = []
        for name, data in groups.items():
            clean_data = data[~np.isnan(data)]
            n = len(clean_data)

            if n < 3:
                results.append(
                    AssumptionResult(
                        name=f"Normality (Shapiro-Wilk) [{name}]",
                        statistic=np.nan,
                        p_value=np.nan,
                        passed=True,  # Default to True with warning due to small sample
                        alpha=self.alpha,
                        details=f"Sample size N={n} is too small for Shapiro-Wilk test (requires N >= 3).",
                    )
                )
                continue

            if np.all(clean_data == clean_data[0]):
                results.append(
                    AssumptionResult(
                        name=f"Normality (Shapiro-Wilk) [{name}]",
                        statistic=1.0,
                        p_value=1.0,
                        passed=True,
                        alpha=self.alpha,
                        details=f"All values in group '{name}' are constant (zero variance).",
                    )
                )
                continue

            stat, p_val = stats.shapiro(clean_data)
            passed = bool(p_val >= self.alpha)
            status_str = "satisfied" if passed else "violated"
            results.append(
                AssumptionResult(
                    name=f"Normality (Shapiro-Wilk) [{name}]",
                    statistic=float(stat),
                    p_value=float(p_val),
                    passed=passed,
                    alpha=self.alpha,
                    details=f"Shapiro-Wilk test W={stat:.4f}, p={p_val:.4f}. Normality {status_str}.",
                )
            )

        return results


class HomogeneityChecker:
    """Evaluates homogeneity of variance (homoscedasticity) using Levene's test."""

    def __init__(self, alpha: float = 0.05, center: str = "median") -> None:
        self.alpha = alpha
        self.center = center

    def check(self, groups: dict[str, np.ndarray]) -> AssumptionResult:
        """Check homogeneity of variance across 2 or more groups.

        Args:
            groups: Dictionary mapping group names to 1D numpy arrays.

        Returns:
            AssumptionResult representing Levene's test outcome.
        """
        clean_groups = [g[~np.isnan(g)] for g in groups.values() if len(g[~np.isnan(g)]) > 0]

        if len(clean_groups) < 2:
            return AssumptionResult(
                name="Homogeneity of Variance (Levene)",
                statistic=np.nan,
                p_value=np.nan,
                passed=True,
                alpha=self.alpha,
                details="Fewer than 2 non-empty groups provided.",
            )

        # Check if all groups have zero variance
        stds = [np.std(g, ddof=1) for g in clean_groups]
        if all(s == 0 for s in stds):
            return AssumptionResult(
                name="Homogeneity of Variance (Levene)",
                statistic=0.0,
                p_value=1.0,
                passed=True,
                alpha=self.alpha,
                details="All groups have zero variance.",
            )

        stat, p_val = stats.levene(*clean_groups, center=self.center)
        passed = bool(p_val >= self.alpha)
        status_str = "satisfied" if passed else "violated"
        return AssumptionResult(
            name="Homogeneity of Variance (Levene)",
            statistic=float(stat),
            p_value=float(p_val),
            passed=passed,
            alpha=self.alpha,
            details=f"Levene's test F={stat:.4f}, p={p_val:.4f}. Equal variances {status_str}.",
        )
