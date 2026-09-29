"""Abstract base class for dataset readers."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataReader(ABC):
    """Abstract DataReader interface for extensible dataset input formats."""

    @abstractmethod
    def read(self, source: Any) -> pd.DataFrame:
        """Read tabular data from a source (filepath, buffer, bytes) into a DataFrame.

        Args:
            source: Path to file or file-like buffer object.

        Returns:
            pd.DataFrame: Loaded tabular raw data.
        """
        pass
