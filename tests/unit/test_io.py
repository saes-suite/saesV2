"""Unit tests for SAES data readers."""

import io

import pytest

from saesV2.domain.exceptions import InputError
from saesV2.io.csv_reader import CSVReader


def test_csv_reader_valid_data() -> None:
    csv_data = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
MOEAD,ZDT1,HV,0,0.665421,True
"""
    reader = CSVReader()
    df = reader.read(io.StringIO(csv_data))
    assert len(df) == 2
    assert list(df.columns) == ["algorithm", "instance", "metric", "run_id", "value", "maximize"]


def test_csv_reader_empty_file() -> None:
    reader = CSVReader()
    with pytest.raises(InputError, match="empty"):
        reader.read(io.StringIO(""))
