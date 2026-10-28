"""Analysis and validation REST endpoints."""

import io
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import saes
from saes.server.schemas import (
    AnalysisResponse,
    CellAnalysisSchema,
    RankingAnalysisResponse,
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
        return ValidationResponse(
            is_valid=report.is_valid,
            error_count=len(report.errors),
            warning_count=len(report.warnings),
            issues=report.issues,
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
                    assumptions=list(res.assumptions),
                    warnings=list(res.warnings),
                    interpretation=res.interpretation,
                    group_summaries=res.group_summaries,
                )
            )

        return AnalysisResponse(cells_count=len(results), results=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Statistical analysis error: {e!s}") from e


@router.post("/ranking", response_model=RankingAnalysisResponse)
async def ranking_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str, Form()],
    alpha: Annotated[float, Form()] = 0.05,
    scope: Annotated[str, Form()] = "all-instances",
    instance: Annotated[str | None, Form()] = None,
    rope_percent: Annotated[float, Form()] = 1.0,
    algo_a: Annotated[str | None, Form()] = None,
    algo_b: Annotated[str | None, Form()] = None,
    bayesian_method: Annotated[str, Form()] = "signed-rank",
) -> RankingAnalysisResponse:
    """Compute Friedman + Nemenyi Critical Distance and Bayesian Simplex ROPE tests on uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
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
