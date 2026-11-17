"""Abstract base class for statistical tests in SAES."""

from abc import ABC, abstractmethod

import numpy as np

from saesV2.domain.models import StatisticalResult


class StatisticalTest(ABC):
    """Abstract Strategy interface for statistical hypothesis testing algorithms."""

    @abstractmethod
    def run(self, groups: dict[str, np.ndarray], alpha: float = 0.05) -> StatisticalResult:
        """Execute statistical test on grouped metric data arrays.

        Args:
            groups: Dictionary mapping group names (e.g. algorithm names) to 1D data arrays.
            alpha: Significance level (default 0.05).

        Returns:
            StatisticalResult: Immutable structured result.
        """
        pass
