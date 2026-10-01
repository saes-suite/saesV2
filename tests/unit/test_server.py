"""Unit tests for SAES FastAPI REST endpoints."""

import io

from fastapi.testclient import TestClient

from saes.server.app import app

client = TestClient(app)

SAMPLE_CSV = """algorithm,instance,metric,run_id,value,maximize
NSGAII,ZDT1,HV,0,0.662145,True
NSGAII,ZDT1,HV,1,0.661890,True
NSGAII,ZDT1,HV,2,0.662012,True
MOEAD,ZDT1,HV,0,0.665421,True
MOEAD,ZDT1,HV,1,0.665100,True
MOEAD,ZDT1,HV,2,0.665900,True
"""


def test_health_endpoint() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_validate_endpoint() -> None:
    file_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res = client.post("/api/v1/validate", files={"file": ("data.csv", file_bytes, "text/csv")})
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert data["error_count"] == 0


def test_analyze_endpoint() -> None:
    file_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res = client.post("/api/v1/analyze", files={"file": ("data.csv", file_bytes, "text/csv")})
    assert res.status_code == 200
    data = res.json()
    assert data["cells_count"] == 1
    assert data["results"][0]["instance"] == "ZDT1"
