"""Analysis and validation REST endpoints."""

import io
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import saes
from saes.server.schemas import (
    AnalysisResponse,
    CellAnalysisSchema,
    ValidationIssueSchema,
    ValidationResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post("/validate", response_model=ValidationResponse)
async def validate_csv(file: Annotated[UploadFile, File(...)]) -> ValidationResponse:
    """Validate uploaded CSV benchmark dataset."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        report = saes.validate(dataset)
        issues = [
            ValidationIssueSchema(
                severity=i.severity.name,
                issue_type=i.issue_type,
                message=i.message,
                column=i.column,
                row_index=i.row_index,
            )
            for i in report.issues
        ]
        return ValidationResponse(
            is_valid=report.is_valid,
            error_count=len(report.errors),
            warning_count=len(report.warnings),
            issues=issues,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {e!s}") from e


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(
    file: Annotated[UploadFile, File(...)],
    alpha: Annotated[float, Form()] = 0.05,
) -> AnalysisResponse:
    """Execute full automated statistical analysis on uploaded CSV dataset."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
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
                    warnings=list(res.warnings),
                )
            )

        return AnalysisResponse(cells_count=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Statistical analysis error: {e!s}") from e
