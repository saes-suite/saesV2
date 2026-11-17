"""Analysis and validation REST endpoints."""

import io
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

import saes
from saes.server.schemas import (
    AnalysisResponse,
    CellAnalysisSchema,
    DataExplorationResponse,
    RankingAnalysisResponse,
    StatisticalTableResponse,
    ValidationResponse,
)
from saes.stats.explorer import export_exploration_table
from saes.stats.tables import export_statistical_table

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


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if not raw or not raw.strip():
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items if items else None


@router.post("/tables", response_model=StatisticalTableResponse)
async def generate_table_csv(
    file: Annotated[UploadFile, File(...)],
    metric: Annotated[str | None, Form()] = None,
    table_type: Annotated[str, Form()] = "instance_algorithm",
    test_type: Annotated[str, Form()] = "auto",
    posthoc_correction: Annotated[str, Form()] = "holm",
    summary_stat: Annotated[str, Form()] = "auto",
    reference_algorithm: Annotated[str | None, Form()] = None,
    scope: Annotated[str, Form()] = "all-instances",
    selected_instance: Annotated[str | None, Form()] = None,
    algorithms_filter: Annotated[str | None, Form()] = None,
    instances_filter: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    precision: Annotated[int, Form()] = 4,
    latex_document_mode: Annotated[str, Form()] = "standalone",
    latex_notation: Annotated[str, Form()] = "scientific",
) -> StatisticalTableResponse:
    """Generate a statistical benchmark table (ANOVA, Friedman, Wilcoxon, Pairwise Matrix) from uploaded CSV."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return saes.generate_table(
            dataset=dataset,
            metric=metric,
            table_type=table_type,
            test_type=test_type,
            posthoc_correction=posthoc_correction,
            summary_stat=summary_stat,
            reference_algorithm=reference_algorithm,
            scope=scope,
            selected_instance=selected_instance,
            algorithms_filter=_parse_csv_list(algorithms_filter),
            instances_filter=_parse_csv_list(instances_filter),
            alpha=alpha,
            precision=precision,
            latex_document_mode=latex_document_mode,
            latex_notation=latex_notation,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table generation error: {e!s}") from e


@router.post("/tables/export")
async def export_table_csv(
    file: Annotated[UploadFile, File(...)],
    export_format: Annotated[str, Form()] = "latex",
    metric: Annotated[str | None, Form()] = None,
    table_type: Annotated[str, Form()] = "instance_algorithm",
    test_type: Annotated[str, Form()] = "auto",
    posthoc_correction: Annotated[str, Form()] = "holm",
    summary_stat: Annotated[str, Form()] = "auto",
    reference_algorithm: Annotated[str | None, Form()] = None,
    scope: Annotated[str, Form()] = "all-instances",
    selected_instance: Annotated[str | None, Form()] = None,
    algorithms_filter: Annotated[str | None, Form()] = None,
    instances_filter: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
    precision: Annotated[int, Form()] = 4,
    latex_document_mode: Annotated[str, Form()] = "standalone",
    latex_notation: Annotated[str, Form()] = "scientific",
    pdf_title: Annotated[str | None, Form()] = None,
    pdf_caption: Annotated[str | None, Form()] = None,
    pdf_theme: Annotated[str, Form()] = "slate",
    pdf_orientation: Annotated[str, Form()] = "landscape",
    pdf_font_size: Annotated[float, Form()] = 8.5,
    pdf_highlight_best: Annotated[bool, Form()] = True,
    pdf_show_zebra: Annotated[bool, Form()] = True,
    pdf_grid_style: Annotated[str, Form()] = "full",
) -> Response:
    """Generate and export a statistical benchmark table in PDF, Word (DOCX), LaTeX, CSV, Markdown, or HTML."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        table_res = saes.generate_table(
            dataset=dataset,
            metric=metric,
            table_type=table_type,
            test_type=test_type,
            posthoc_correction=posthoc_correction,
            summary_stat=summary_stat,
            reference_algorithm=reference_algorithm,
            scope=scope,
            selected_instance=selected_instance,
            algorithms_filter=_parse_csv_list(algorithms_filter),
            instances_filter=_parse_csv_list(instances_filter),
            alpha=alpha,
            precision=precision,
            latex_document_mode=latex_document_mode,
            latex_notation=latex_notation,
        )
        file_bytes, media_type, ext = export_statistical_table(
            table_res,
            export_format=export_format,
            document_mode=latex_document_mode,
            notation=latex_notation,
            title=pdf_title,
            caption=pdf_caption,
            pdf_theme=pdf_theme,
            pdf_orientation=pdf_orientation,
            pdf_font_size=pdf_font_size,
            pdf_highlight_best=pdf_highlight_best,
            pdf_show_zebra=pdf_show_zebra,
            pdf_grid_style=pdf_grid_style,
        )
        out_name = f"saes_table_{table_res.metric}_{table_res.table_type}.{ext}"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table export error: {e!s}") from e


@router.post("/explore", response_model=DataExplorationResponse)
async def explore_csv(
    file: Annotated[UploadFile, File(...)],
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> DataExplorationResponse:
    """Explore dataset observations, reveal multi-metric pivots, check normality, and get test recommendations."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        return saes.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data exploration error: {e!s}") from e


@router.post("/explore/export")
async def export_explore_csv(
    file: Annotated[UploadFile, File(...)],
    export_format: Annotated[str, Form()] = "csv",
    view_mode: Annotated[str, Form()] = "reveal_metrics",
    stat_format: Annotated[str, Form()] = "mean_std",
    normality_filter: Annotated[str, Form()] = "all",
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> Response:
    """Export an Observations & Data exploration view in PDF, Word (DOCX), LaTeX, CSV, Markdown, or HTML."""
    content = await file.read()
    try:
        dataset = saes.load(io.BytesIO(content))
        exp_res = saes.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
        file_bytes, media_type, ext = export_exploration_table(
            exploration=exp_res,
            view_mode=view_mode,
            export_format=export_format,
            stat_format=stat_format,
            normality_filter=normality_filter,
        )
        out_name = f"saes_observations_{view_mode}.{ext}"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Observation export error: {e!s}") from e
