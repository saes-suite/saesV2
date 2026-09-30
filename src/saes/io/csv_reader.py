"""CSV data reader implementation for SAES input files."""

from typing import Any

import pandas as pd

from saes.domain.exceptions import InputError
from saes.io.base import DataReader


class CSVReader(DataReader):
    """Reads comma-separated value (CSV) files compliant with SAES specification."""

    def read(self, source: Any) -> pd.DataFrame:
        """Read CSV data from file path or buffer.

        Args:
            source: Path string, Path object, or file-like buffer.

        Returns:
            pd.DataFrame: Unvalidated raw pandas DataFrame.

        Raises:
            InputError: If file cannot be parsed or read.
        """
        try:
            df = pd.read_csv(source, skipinitialspace=True)
            if df.empty:
                raise InputError("Input CSV file is empty.")
            for col in ("algorithm", "instance", "metric"):
                if col in df.columns:
                    df[col] = df[col].astype(str)
            return df
        except pd.errors.EmptyDataError as e:
            raise InputError("Input CSV file is empty.") from e
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"Failed to read CSV input: {str(e)}") from e
