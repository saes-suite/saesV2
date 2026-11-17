"""Storage utilities for saved benchmark datasets."""

import os
from pathlib import Path


def get_datasets_dir() -> Path:
    """Return the absolute directory path where uploaded benchmark datasets are persisted.

    Can be customized via the SAES_DATASETS_DIR environment variable.
    Defaults to the 'datasets' folder in the repository root.
    """
    custom_dir = os.getenv("SAES_DATASETS_DIR")
    if custom_dir:
        path = Path(custom_dir).resolve()
    else:
        # Default to 'datasets' folder in the saes repository root
        path = (Path(__file__).resolve().parents[3] / "datasets").resolve()

    path.mkdir(parents=True, exist_ok=True)
    return path
