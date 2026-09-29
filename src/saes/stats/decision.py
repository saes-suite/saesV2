"""Statistical decision tree engine for SAES."""

import numpy as np
import pandas as pd

from saes.domain.exceptions import StatisticalAnalysisError
from saes.domain.models import StatisticalResult
from saes.stats.anova import OneWayANOVA, WelchANOVA
from saes.stats.assumptions import HomogeneityChecker
from saes.stats.base import StatisticalTest
from saes.stats.ttest import IndependentTTest, PairedTTest, WelchTTest


def is_paired_data(cell_df: pd.DataFrame) -> bool:
    """Check if observations across algorithms are matched 1-to-1 by run_id."""
    if "run_id" not in cell_df.columns or "algorithm" not in cell_df.columns:
        return False
    pivoted = cell_df.pivot(index="run_id", columns="algorithm", values="value")
    # If no missing values after pivoting, each run_id has exactly 1 observation per algorithm
    return bool(not pivoted.isnull().any().any() and len(pivoted) >= 2)


class TestDecisionEngine:
    """Automated statistical test selection decision engine."""

    __test__ = False

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def select_test(self, groups: dict[str, np.ndarray], is_paired: bool = False) -> StatisticalTest:
        """Select the appropriate statistical test based on mathematical decision rules.

        Args:
            groups: Dictionary mapping group names to 1D data arrays.
            is_paired: Whether observations are paired across groups.

        Returns:
            StatisticalTest strategy instance.
        """
        k = len(groups)
        if k < 2:
            raise StatisticalAnalysisError(f"Statistical test selection requires >= 2 groups, got {k}.")

        if k == 2:
            if is_paired:
                return PairedTTest()
            homogeneity = HomogeneityChecker(alpha=self.alpha).check(groups)
            if homogeneity.passed:
                return IndependentTTest()
            else:
                return WelchTTest()
        else:
            homogeneity = HomogeneityChecker(alpha=self.alpha).check(groups)
            if homogeneity.passed:
                return OneWayANOVA()
            else:
                return WelchANOVA()

    def analyze_cell(self, cell_df: pd.DataFrame, force_test: StatisticalTest | None = None) -> StatisticalResult:
        """Analyze a single (instance, metric) DataFrame cell using automated decision engine or forced test.

        Args:
            cell_df: Filtered DataFrame for a single (instance, metric) pair.
            force_test: Optional explicit StatisticalTest instance to override decision engine.

        Returns:
            StatisticalResult: Outcome of statistical evaluation.
        """
        groups: dict[str, np.ndarray] = {}
        for alg, grp in cell_df.groupby("algorithm"):
            groups[str(alg)] = grp["value"].to_numpy(dtype=float)

        if force_test is not None:
            return force_test.run(groups, alpha=self.alpha)

        paired = is_paired_data(cell_df)
        selected_test = self.select_test(groups, is_paired=paired)
        return selected_test.run(groups, alpha=self.alpha)
