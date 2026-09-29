"""Unit tests for DatasetValidator."""

import pandas as pd
import pytest

from saes.domain.exceptions import ValidationError
from saes.validation.validator import DatasetValidator


def test_validator_valid_dataset() -> None:
    df = pd.DataFrame(
        {
            "algorithm": ["NSGAII", "NSGAII", "MOEAD", "MOEAD"],
            "instance": ["ZDT1", "ZDT1", "ZDT1", "ZDT1"],
            "metric": ["HV", "HV", "HV", "HV"],
            "run_id": [0, 1, 0, 1],
            "value": [0.66, 0.67, 0.68, 0.69],
            "maximize": [True, True, True, True],
        }
    )
    validator = DatasetValidator()
    report = validator.validate(df)
    assert report.is_valid
    assert len(report.errors) == 0


def test_validator_missing_column() -> None:
    df = pd.DataFrame(
        {
            "algorithm": ["NSGAII"],
            "instance": ["ZDT1"],
            # missing metric, run_id, value, maximize
        }
    )
    validator = DatasetValidator()
    report = validator.validate(df)
    assert not report.is_valid
    assert any(err.issue_type == "MissingColumns" for err in report.errors)


def test_validator_invalid_types_and_missing_values() -> None:
    df = pd.DataFrame(
        {
            "algorithm": ["NSGAII", "MOEAD"],
            "instance": ["ZDT1", "ZDT1"],
            "metric": ["HV", None],
            "run_id": [-1, 0],
            "value": ["not_a_number", 0.65],
            "maximize": [True, "invalid_bool"],
        }
    )
    validator = DatasetValidator()
    report = validator.validate(df)
    assert not report.is_valid
    error_types = {e.issue_type for e in report.errors}
    assert "MissingValues" in error_types or "InvalidType" in error_types
    assert "InvalidValue" in error_types or "InvalidType" in error_types


def test_validator_validate_or_raise() -> None:
    df = pd.DataFrame({"invalid": [1, 2, 3]})
    validator = DatasetValidator()
    with pytest.raises(ValidationError):
        validator.validate_or_raise(df)
