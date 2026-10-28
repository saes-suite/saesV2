import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import saes
from saes.io.csv_reader import CSVReader
from saes.server.schemas import (
    AnalysisResponse,
    CellAnalysisSchema,
    DatasetDetailSchema,
    DatasetItemSchema,
    DatasetUpdateSchema,
    RankingAnalysisResponse,
    ValidationResponse,
)
from saes.server.storage import get_datasets_dir
from saes.validation.validator import DatasetValidator

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


def _count_csv_rows(file_path: Path) -> int:
    """Quickly count non-empty data rows in a CSV file, excluding header."""
    try:
        count = 0
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    count += 1
        return max(0, count - 1)
    except Exception:
        return 0


def _build_dataset_item(file_path: Path) -> DatasetItemSchema:
    """Build metadata item for a dataset file."""
    stat = file_path.stat()
    return DatasetItemSchema(
        name=file_path.stem,
        filename=file_path.name,
        path=str(file_path),
        size_bytes=stat.st_size,
        row_count=_count_csv_rows(file_path),
        modified_at=stat.st_mtime,
    )


@router.get("", response_model=list[DatasetItemSchema])
def list_datasets() -> list[DatasetItemSchema]:
    """List all saved benchmark datasets in the backend storage directory."""
    datasets_dir = get_datasets_dir()
    items: list[DatasetItemSchema] = []
    for p in sorted(datasets_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            items.append(_build_dataset_item(p))
    return items


@router.post("/upload", response_model=DatasetItemSchema)
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> DatasetItemSchema:
    """Upload, validate, and save a CSV dataset file into backend storage (gitignore-backed)."""
    filename = Path(file.filename or "uploaded_benchmark.csv").name
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Strict structural and schema validation before accepting upload
    try:
        df = CSVReader().read(io.BytesIO(content))
        report = DatasetValidator().validate(df)
        missing_col_errors = [e for e in report.errors if e.issue_type == "MissingColumns"]
        if missing_col_errors:
            error_msgs = "; ".join(e.message for e in missing_col_errors)
            raise HTTPException(
                status_code=422,
                detail=f"Dataset validation failed: {error_msgs}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV dataset: {e!s}") from e

    datasets_dir = get_datasets_dir()
    target_path = datasets_dir / filename
    target_path.write_bytes(content)
    return _build_dataset_item(target_path)


@router.put("/{filename}", response_model=DatasetDetailSchema)
def update_dataset(filename: str, payload: DatasetUpdateSchema) -> DatasetDetailSchema:
    """Save or update CSV dataset content directly in backend storage."""
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    file_path = get_datasets_dir() / safe_name
    try:
        file_path.write_text(payload.content, encoding="utf-8")
        stat = file_path.stat()
        return DatasetDetailSchema(
            name=file_path.stem,
            filename=file_path.name,
            path=str(file_path),
            size_bytes=stat.st_size,
            row_count=_count_csv_rows(file_path),
            content=payload.content,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving dataset: {e!s}") from e


@router.get("/{filename}", response_model=DatasetDetailSchema)
def get_dataset(filename: str) -> DatasetDetailSchema:
    """Retrieve full content and metadata for a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        stat = file_path.stat()
        return DatasetDetailSchema(
            name=file_path.stem,
            filename=file_path.name,
            path=str(file_path),
            size_bytes=stat.st_size,
            row_count=_count_csv_rows(file_path),
            content=content,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {e!s}") from e


@router.post("/{filename}/validate", response_model=ValidationResponse)
def validate_dataset_from_path(filename: str) -> ValidationResponse:
    """Validate a saved benchmark dataset by loading directly from its server path."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        report = saes.validate(dataset)
        return ValidationResponse(
            is_valid=report.is_valid,
            error_count=len(report.errors),
            warning_count=len(report.warnings),
            issues=report.issues,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset validation error: {e!s}") from e


@router.post("/{filename}/analyze", response_model=AnalysisResponse)
def analyze_dataset_from_path(
    filename: str,
    alpha: Annotated[float, Form()] = 0.05,
) -> AnalysisResponse:
    """Execute statistical analysis on a saved dataset by loading directly from its server path."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        report = saes.analyze(dataset, alpha=alpha, generate_plots=False)

        results: list[CellAnalysisSchema] = []
        for cell in report.cell_analyses:
            res = cell.result
            results.append(
                CellAnalysisSchema(
                    instance=cell.instance,
                    metric=cell.metric,
                    test_name=res.test_name,
                    statistic=res.statistic,
                    p_value=res.p_value,
                    degrees_of_freedom=res.degrees_of_freedom,
                    is_significant=res.is_significant,
                    effect_size_name=res.effect_size.name if res.effect_size else None,
                    effect_size_value=res.effect_size.value if res.effect_size else None,
                    effect_size_magnitude=res.effect_size.magnitude if res.effect_size else None,
                    assumptions=list(res.assumptions),
                    warnings=list(res.warnings),
                    interpretation=res.interpretation,
                    group_summaries=res.group_summaries,
                )
            )

        return AnalysisResponse(cells_count=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Statistical analysis error: {e!s}") from e


@router.post("/{filename}/ranking", response_model=RankingAnalysisResponse)
def ranking_dataset_from_path(
    filename: str,
    metric: Annotated[str, Form()],
    alpha: Annotated[float, Form()] = 0.05,
    scope: Annotated[str, Form()] = "all-instances",
    instance: Annotated[str | None, Form()] = None,
    rope_percent: Annotated[float, Form()] = 1.0,
    algo_a: Annotated[str | None, Form()] = None,
    algo_b: Annotated[str | None, Form()] = None,
    bayesian_method: Annotated[str, Form()] = "signed-rank",
) -> RankingAnalysisResponse:
    """Compute Friedman + Nemenyi CD and Bayesian Simplex ROPE tests on a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saes.load(file_path)
        cd_res = saes.statistics.friedman_nemenyi(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            scope=scope,
            instance=instance,
        )
        bayes_sign_res = saes.statistics.bayesian_sign(
            dataset=dataset,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            algo_a=algo_a,
            algo_b=algo_b,
        )
        bayes_sr_res = saes.statistics.bayesian_signed_rank(
            dataset=dataset,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            algo_a=algo_a,
            algo_b=algo_b,
        )
        active_bayes = (
            bayes_sign_res
            if bayesian_method.strip().lower() in {"sign", "bayesian-sign", "sign-test"}
            else bayes_sr_res
        )
        return RankingAnalysisResponse(
            critical_distance=cd_res,
            bayesian_ranking=active_bayes,
            bayesian_sign=bayes_sign_res,
            bayesian_signed_rank=bayes_sr_res,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ranking analysis error: {e!s}") from e


@router.delete("/{filename}")
def delete_dataset(filename: str) -> dict[str, str]:
    """Delete a saved dataset from the backend storage directory."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        file_path.unlink()
        return {"status": "deleted", "filename": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {e!s}") from e
