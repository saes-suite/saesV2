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


def test_chat_endpoint() -> None:
    res = client.post(
        "/api/v1/chat",
        json={
            "query": "Why was this test chosen?",
            "csv_data": SAMPLE_CSV,
            "alpha": 0.05,
            "provider": "offline",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert data["context_used"] is True


def test_get_local_models_endpoint() -> None:
    res = client.get("/api/v1/chat/models/local?base_url=http://127.0.0.1:99999")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert data["status"] in ("ok", "empty")


def test_probe_provider_endpoint() -> None:
    # 1. Offline probe
    res_off = client.post("/api/v1/chat/probe", json={"provider": "offline"})
    assert res_off.status_code == 200
    data_off = res_off.json()
    assert data_off["status"] == "ok"
    assert "deterministic-statistical-engine" in data_off["models"]

    # 2. Local probe (unreachable port)
    res_local = client.post(
        "/api/v1/chat/probe",
        json={"provider": "local", "base_url": "http://127.0.0.1:99999"},
    )
    assert res_local.status_code == 200
    data_local = res_local.json()
    assert data_local["status"] in ("not_installed", "binary_found_not_running", "error", "ok")

    # 3. Gemini missing key
    res_gem = client.post("/api/v1/chat/probe", json={"provider": "gemini", "api_key": ""})
    assert res_gem.status_code == 200
    assert res_gem.json()["status"] == "error"


def test_start_ollama_daemon_endpoint() -> None:
    res = client.post("/api/v1/chat/ollama/start")
    assert res.status_code == 200
    assert "started" in res.json()


def test_pull_ollama_endpoint() -> None:
    res = client.post(
        "/api/v1/chat/ollama/pull",
        json={"name": "llama3.2", "base_url": "http://127.0.0.1:99999"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "error"


def test_pull_ollama_stream_endpoint() -> None:
    res = client.post(
        "/api/v1/chat/ollama/pull/stream",
        json={"name": "llama3.2", "base_url": "http://127.0.0.1:99999"},
    )
    assert res.status_code == 200
    assert "error" in res.text


