"""Unit tests for SAES dataset persistence and path-based analysis endpoints."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from saesV2.server.app import app

client = TestClient(app)

SAMPLE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""


@pytest.fixture(autouse=True)
def setup_teardown_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate datasets storage to temporary directory for unit testing."""
    monkeypatch.setenv("SAES_DATASETS_DIR", str(tmp_path))
    yield
    # Cleaned up automatically by tmp_path


def test_datasets_crud_and_path_analysis():
    # 1. Initially empty
    res = client.get("/api/v1/datasets")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Upload dataset
    file_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    upload_res = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test_benchmark.csv", file_bytes, "text/csv")},
    )
    assert upload_res.status_code == 200
    item = upload_res.json()
    assert item["name"] == "test_benchmark"
    assert item["filename"] == "test_benchmark.csv"
    assert item["row_count"] == 6

    # 3. List datasets contains uploaded item
    list_res = client.get("/api/v1/datasets")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    # 4. Get dataset details
    detail_res = client.get("/api/v1/datasets/test_benchmark.csv")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "algorithm,instance,metric" in detail["content"]
    assert detail["row_count"] == 6

    # 5. Validate dataset directly from server path
    val_res = client.post("/api/v1/datasets/test_benchmark.csv/validate")
    assert val_res.status_code == 200
    assert val_res.json()["is_valid"] is True

    # 6. Analyze dataset directly from server path
    ana_res = client.post("/api/v1/datasets/test_benchmark.csv/analyze", data={"alpha": 0.05})
    assert ana_res.status_code == 200
    ana_data = ana_res.json()
    assert ana_data["cells_count"] == 1
    assert ana_data["results"][0]["instance"] == "ZDT1"

    # 7. Chat with dataset_name
    chat_res = client.post(
        "/api/v1/chat",
        json={"query": "Summarize findings", "dataset_name": "test_benchmark.csv", "alpha": 0.05},
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["context_used"] is True

    # 8. Update dataset content via PUT
    updated_csv = SAMPLE_CSV + "MOEAD,ZDT1,HV,3,0.666000,True\n"
    put_res = client.put(
        "/api/v1/datasets/test_benchmark.csv",
        json={"content": updated_csv},
    )
    assert put_res.status_code == 200
    assert put_res.json()["row_count"] == 7

    # 9. Delete dataset
    del_res = client.delete("/api/v1/datasets/test_benchmark.csv")
    assert del_res.status_code == 200
    assert client.get("/api/v1/datasets").json() == []


def test_dataset_not_found_handling():
    res = client.get("/api/v1/datasets/non_existent.csv")
    assert res.status_code == 404

    val_res = client.post("/api/v1/datasets/non_existent.csv/validate")
    assert val_res.status_code == 404

    ana_res = client.post("/api/v1/datasets/non_existent.csv/analyze")
    assert ana_res.status_code == 404


def test_upload_dataset_rejects_invalid_schema():
    # Missing 'value' column
    bad_csv = "algorithm,instance,metric,run_id,maximize\nNSGAII,ZDT1,HV,0,True\n"
    res = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("corrupt.csv", io.BytesIO(bad_csv.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 422
    assert "Missing required column(s)" in res.json()["detail"]
    # Verify file was not written to storage
    assert client.get("/api/v1/datasets").json() == []
