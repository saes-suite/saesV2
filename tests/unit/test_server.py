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
    assert "interpretation" in data["results"][0]
    assert "group_summaries" in data["results"][0]


def test_ranking_endpoint() -> None:
    file_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res = client.post(
        "/api/v1/ranking",
        files={"file": ("data.csv", file_bytes, "text/csv")},
        data={
            "metric": "HV",
            "alpha": "0.05",
            "scope": "single-instance",
            "instance": "ZDT1",
            "algo_a": "NSGAII",
            "algo_b": "MOEAD",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["critical_distance"]["metric"] == "HV"
    assert data["critical_distance"]["k"] == 2
    assert data["bayesian_ranking"]["reference_algo"] == "MOEAD"
    assert data["bayesian_ranking"]["algo_a"] == "NSGAII"
    assert data["bayesian_ranking"]["algo_b"] == "MOEAD"
    assert len(data["bayesian_ranking"]["points_x"]) > 0
    assert len(data["bayesian_ranking"]["boundary_lines"]) == 3
    assert data["bayesian_sign"]["bayesian_method"] == "sign"
    assert data["bayesian_signed_rank"]["bayesian_method"] == "signed-rank"


def test_tables_and_explore_endpoints() -> None:
    file_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_tbl = client.post(
        "/api/v1/tables",
        files={"file": ("data.csv", file_bytes, "text/csv")},
        data={
            "metric": "HV",
            "table_type": "instance_algorithm",
            "test_type": "wilcoxon",
            "posthoc_correction": "holm",
            "alpha": "0.05",
        },
    )
    assert res_tbl.status_code == 200
    tbl_data = res_tbl.json()
    assert tbl_data["metric"] == "HV"
    assert len(tbl_data["instance_rows"]) == 1
    assert len(tbl_data["pairwise_matrix"]) == 0
    assert "\\begin{table*}" in tbl_data["latex_preview"]

    # Test pairwise_matrix table type
    pw_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_pw = client.post(
        "/api/v1/tables",
        files={"file": ("data.csv", pw_bytes, "text/csv")},
        data={
            "metric": "HV",
            "table_type": "pairwise_matrix",
            "test_type": "wilcoxon",
            "posthoc_correction": "holm",
            "alpha": "0.05",
        },
    )
    assert res_pw.status_code == 200
    pw_json = res_pw.json()
    assert len(pw_json["pairwise_matrix"]) == 2
    assert "\\documentclass{article}" in pw_json["latex_preview"]
    assert "1vs1 Table" in pw_json["latex_preview"]
    assert len(pw_json["pairwise_matrix"][0][1]["instance_symbols"]) == 1

    # Test export endpoint (docx and pdf)
    for fmt in ["latex", "csv", "docx", "pdf"]:
        exp_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
        res_exp = client.post(
            "/api/v1/tables/export",
            files={"file": ("data.csv", exp_bytes, "text/csv")},
            data={"metric": "HV", "export_format": fmt},
        )
        assert res_exp.status_code == 200
        assert len(res_exp.content) > 20

    # Test PDF export with presentation & styling parameters
    exp_pdf_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_pdf_styled = client.post(
        "/api/v1/tables/export",
        files={"file": ("data.csv", exp_pdf_bytes, "text/csv")},
        data={
            "metric": "HV",
            "export_format": "pdf",
            "pdf_title": "Benchmark Report 2026",
            "pdf_caption": "Table styling verification test",
            "pdf_theme": "navy",
            "pdf_orientation": "portrait",
            "pdf_font_size": "9.0",
            "pdf_highlight_best": "true",
            "pdf_show_zebra": "true",
            "pdf_grid_style": "minimal",
        },
    )
    assert res_pdf_styled.status_code == 200
    assert len(res_pdf_styled.content) > 50
    assert res_pdf_styled.headers["content-type"] == "application/pdf"

    # Test explore endpoint and explore/export endpoint
    exp_file = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_explore = client.post(
        "/api/v1/explore",
        files={"file": ("data.csv", exp_file, "text/csv")},
        data={"alpha": "0.05"},
    )
    assert res_explore.status_code == 200
    exp_data = res_explore.json()
    assert exp_data["context_diagnostics"]["k_algorithms"] == 2
    assert len(exp_data["recommendations"]) >= 2
    assert len(exp_data["distribution_profiles"]) == 2

    for fmt in ["latex", "csv", "docx", "pdf"]:
        exp_obs_bytes = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
        res_obs_exp = client.post(
            "/api/v1/explore/export",
            files={"file": ("data.csv", exp_obs_bytes, "text/csv")},
            data={"export_format": fmt, "view_mode": "reveal_metrics"},
        )
        assert res_obs_exp.status_code == 200
        assert len(res_obs_exp.content) > 20

    # Test Profiles endpoint
    prof_file = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_prof = client.post("/api/v1/profiles", files={"file": ("data.csv", prof_file, "text/csv")})
    assert res_prof.status_code == 200
    prof_data = res_prof.json()
    assert len(prof_data["curves"]) == 2
    assert len(prof_data["meta_rankings"]) == 2

    # Test Power endpoint
    power_file = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_power = client.post("/api/v1/power", files={"file": ("data.csv", power_file, "text/csv")})
    assert res_power.status_code == 200
    power_data = res_power.json()
    assert len(power_data["sample_size_recommendations"]) >= 3
    assert len(power_data["power_curves"]) >= 3

    # Test Reproducibility endpoint
    rep_file = io.BytesIO(SAMPLE_CSV.encode("utf-8"))
    res_rep = client.post("/api/v1/reproducibility", files={"file": ("data.csv", rep_file, "text/csv")})
    assert res_rep.status_code == 200
    rep_data = res_rep.json()
    assert "import saes" in rep_data["python_script"]
    assert "nbformat" in rep_data["jupyter_notebook_json"]

