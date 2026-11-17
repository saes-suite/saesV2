import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile

import saesV2
from saesV2.io.csv_reader import CSVReader
from saesV2.server.schemas import (
    AnalysisResponse,
    CellAnalysisSchema,
    DataExplorationResponse,
    DatasetDetailSchema,
    DatasetItemSchema,
    DatasetUpdateSchema,
    PerformanceProfilesResponse,
    PowerAnalysisResponse,
    RankingAnalysisResponse,
    ReproducibilityResponse,
    StatisticalTableResponse,
    ValidationResponse,
)
from saesV2.server.storage import get_datasets_dir
from saesV2.stats.explorer import export_exploration_table
from saesV2.stats.tables import export_statistical_table
from saesV2.validation.validator import DatasetValidator

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
        dataset = saesV2.load(file_path)
        report = saesV2.validate(dataset)
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
        dataset = saesV2.load(file_path)
        report = saesV2.analyze(dataset, alpha=alpha, generate_plots=False)

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
        dataset = saesV2.load(file_path)
        cd_res = saesV2.statistics.friedman_nemenyi(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            scope=scope,
            instance=instance,
        )
        bayes_sign_res = saesV2.statistics.bayesian_sign(
            dataset=dataset,
            metric=metric,
            scope=scope,
            instance=instance,
            rope_percent=rope_percent,
            algo_a=algo_a,
            algo_b=algo_b,
        )
        bayes_sr_res = saesV2.statistics.bayesian_signed_rank(
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


@router.get("/{filename}/tables", response_model=StatisticalTableResponse)
def get_dataset_table(
    filename: str,
    response: Response,
    metric: Annotated[str | None, Query()] = None,
    table_type: Annotated[str, Query()] = "instance_algorithm",
    test_type: Annotated[str, Query()] = "auto",
    posthoc_correction: Annotated[str, Query()] = "holm",
    summary_stat: Annotated[str, Query()] = "auto",
    reference_algorithm: Annotated[str | None, Query()] = None,
    scope: Annotated[str, Query()] = "all-instances",
    selected_instance: Annotated[str | None, Query()] = None,
    algorithms_filter: Annotated[str | None, Query()] = None,
    instances_filter: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    precision: Annotated[int, Query()] = 4,
) -> StatisticalTableResponse:
    """Generate a statistical benchmark table via HTTP GET (cacheable by standard HTTP clients)."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        response.headers["Cache-Control"] = "public, max-age=3600"
        return saesV2.generate_table(
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
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table generation error: {e!s}") from e


@router.post("/{filename}/tables", response_model=StatisticalTableResponse)
def generate_dataset_table(
    filename: str,
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
    """Generate a statistical benchmark table (ANOVA, Friedman, Wilcoxon, Pairwise) for a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        return saesV2.generate_table(
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


@router.get("/{filename}/tables/export")
def export_dataset_table_get(
    filename: str,
    export_format: Annotated[str, Query()] = "latex",
    metric: Annotated[str | None, Query()] = None,
    table_type: Annotated[str, Query()] = "instance_algorithm",
    test_type: Annotated[str, Query()] = "auto",
    posthoc_correction: Annotated[str, Query()] = "holm",
    summary_stat: Annotated[str, Query()] = "auto",
    reference_algorithm: Annotated[str | None, Query()] = None,
    scope: Annotated[str, Query()] = "all-instances",
    selected_instance: Annotated[str | None, Query()] = None,
    algorithms_filter: Annotated[str | None, Query()] = None,
    instances_filter: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    precision: Annotated[int, Query()] = 4,
    latex_document_mode: Annotated[str, Query()] = "standalone",
    latex_notation: Annotated[str, Query()] = "scientific",
    pdf_title: Annotated[str | None, Query()] = None,
    pdf_caption: Annotated[str | None, Query()] = None,
    pdf_theme: Annotated[str, Query()] = "slate",
    pdf_orientation: Annotated[str, Query()] = "landscape",
    pdf_font_size: Annotated[float, Query()] = 8.5,
    pdf_highlight_best: Annotated[bool, Query()] = True,
    pdf_show_zebra: Annotated[bool, Query()] = True,
    pdf_grid_style: Annotated[str, Query()] = "full",
) -> Response:
    """Export a statistical benchmark table via HTTP GET."""
    return export_dataset_table(
        filename=filename,
        export_format=export_format,
        metric=metric,
        table_type=table_type,
        test_type=test_type,
        posthoc_correction=posthoc_correction,
        summary_stat=summary_stat,
        reference_algorithm=reference_algorithm,
        scope=scope,
        selected_instance=selected_instance,
        algorithms_filter=algorithms_filter,
        instances_filter=instances_filter,
        alpha=alpha,
        precision=precision,
        latex_document_mode=latex_document_mode,
        latex_notation=latex_notation,
        pdf_title=pdf_title,
        pdf_caption=pdf_caption,
        pdf_theme=pdf_theme,
        pdf_orientation=pdf_orientation,
        pdf_font_size=pdf_font_size,
        pdf_highlight_best=pdf_highlight_best,
        pdf_show_zebra=pdf_show_zebra,
        pdf_grid_style=pdf_grid_style,
    )


@router.post("/{filename}/tables/export")
def export_dataset_table(
    filename: str,
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
    """Generate and export a statistical benchmark table for a saved dataset in PDF, DOCX, LaTeX, CSV, MD, or HTML."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        table_res = saesV2.generate_table(
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
            headers={
                "Content-Disposition": f'attachment; filename="{out_name}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Table export error: {e!s}") from e


@router.get("/{filename}/explore", response_model=DataExplorationResponse)
def get_explore_dataset_from_path(
    filename: str,
    response: Response,
    metrics: Annotated[str | None, Query()] = None,
    algorithms: Annotated[str | None, Query()] = None,
    instances: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
) -> DataExplorationResponse:
    """Explore dataset observations via HTTP GET (cacheable by standard HTTP clients)."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        response.headers["Cache-Control"] = "public, max-age=3600"
        return saesV2.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data exploration error: {e!s}") from e


@router.post("/{filename}/explore", response_model=DataExplorationResponse)
def explore_dataset_from_path(
    filename: str,
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> DataExplorationResponse:
    """Explore observations, reveal multi-metric pivots, check normality, and recommend tests for a saved dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        return saesV2.explore(
            dataset=dataset,
            metrics=_parse_csv_list(metrics),
            algorithms=_parse_csv_list(algorithms),
            instances=_parse_csv_list(instances),
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data exploration error: {e!s}") from e


@router.get("/{filename}/explore/export")
def export_explore_dataset_get(
    filename: str,
    export_format: Annotated[str, Query()] = "csv",
    view_mode: Annotated[str, Query()] = "reveal_metrics",
    stat_format: Annotated[str, Query()] = "mean_std",
    normality_filter: Annotated[str, Query()] = "all",
    metrics: Annotated[str | None, Query()] = None,
    algorithms: Annotated[str | None, Query()] = None,
    instances: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
) -> Response:
    """Export an Observations & Data exploration view via HTTP GET."""
    return export_explore_dataset(
        filename=filename,
        export_format=export_format,
        view_mode=view_mode,
        stat_format=stat_format,
        normality_filter=normality_filter,
        metrics=metrics,
        algorithms=algorithms,
        instances=instances,
        alpha=alpha,
    )


@router.post("/{filename}/explore/export")
def export_explore_dataset(
    filename: str,
    export_format: Annotated[str, Form()] = "csv",
    view_mode: Annotated[str, Form()] = "reveal_metrics",
    stat_format: Annotated[str, Form()] = "mean_std",
    normality_filter: Annotated[str, Form()] = "all",
    metrics: Annotated[str | None, Form()] = None,
    algorithms: Annotated[str | None, Form()] = None,
    instances: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.05,
) -> Response:
    """Export an Observations & Data exploration view for a saved dataset in PDF, DOCX, LaTeX, CSV, MD, or HTML."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        exp_res = saesV2.explore(
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
            headers={
                "Content-Disposition": f'attachment; filename="{out_name}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Observation export error: {e!s}") from e


@router.get("/{filename}/profiles", response_model=PerformanceProfilesResponse)
@router.post("/{filename}/profiles", response_model=PerformanceProfilesResponse)
def get_dataset_profiles(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    max_tau: Annotated[float | None, Query()] = None,
) -> PerformanceProfilesResponse:
    """Compute Dolan-Moré performance profiles and global Win/Tie/Loss matrix for a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        return saesV2.compute_performance_profiles(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            max_tau=max_tau,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance profile error: {e!s}") from e


@router.get("/{filename}/power", response_model=PowerAnalysisResponse)
@router.post("/{filename}/power", response_model=PowerAnalysisResponse)
def get_dataset_power(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
    target_power: Annotated[float, Query()] = 0.80,
    target_effect_size: Annotated[float, Query()] = 0.50,
) -> PowerAnalysisResponse:
    """Compute statistical power analysis and experimental sample size planning for a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        dataset = saesV2.load(file_path)
        return saesV2.analyze_statistical_power(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            target_power=target_power,
            target_effect_size=target_effect_size,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Power analysis error: {e!s}") from e


@router.get("/{filename}/reproducibility", response_model=ReproducibilityResponse)
@router.post("/{filename}/reproducibility", response_model=ReproducibilityResponse)
def get_dataset_reproducibility(
    filename: str,
    metric: Annotated[str | None, Query()] = None,
    alpha: Annotated[float, Query()] = 0.05,
) -> ReproducibilityResponse:
    """Generate reproducible Python script, Jupyter notebook, and FAIR manifest for a stored dataset."""
    safe_name = Path(filename).name
    file_path = get_datasets_dir() / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        dataset = saesV2.load(file_path)
        return saesV2.generate_reproducibility_artifacts(
            dataset=dataset,
            metric=metric,
            alpha=alpha,
            filename=safe_name,
            raw_csv=raw_text,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reproducibility error: {e!s}") from e


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
