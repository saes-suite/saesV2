"""Application pipeline orchestrating dataset loading, validation, analysis, and plotting."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from saesV2.domain.models import Dataset, StatisticalResult, ValidationReport
from saesV2.io.csv_reader import CSVReader
from saesV2.stats.base import StatisticalTest
from saesV2.stats.decision import TestDecisionEngine
from saesV2.validation.validator import DatasetValidator


@dataclass
class CellAnalysis:
    """Analysis summary for a single (instance, metric) experimental context."""

    instance: str
    metric: str
    result: StatisticalResult
    figure: Any | None = None


@dataclass
class AnalysisReport:
    """Full analysis report across all experimental cells in a dataset."""

    cell_analyses: list[CellAnalysis] = field(default_factory=list)

    def get(self, instance: str, metric: str) -> CellAnalysis | None:
        """Fetch cell analysis by instance and metric name."""
        for ca in self.cell_analyses:
            if ca.instance == instance and ca.metric == metric:
                return ca
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all statistical results into a flat pandas summary DataFrame."""
        rows: list[dict[str, Any]] = []
        for ca in self.cell_analyses:
            res = ca.result
            row = {
                "instance": ca.instance,
                "metric": ca.metric,
                "test_name": res.test_name,
                "statistic": res.statistic,
                "p_value": res.p_value,
                "degrees_of_freedom": str(res.degrees_of_freedom),
                "is_significant": res.is_significant,
                "effect_size_name": res.effect_size.name if res.effect_size else "",
                "effect_size_value": res.effect_size.value if res.effect_size else None,
                "effect_size_magnitude": res.effect_size.magnitude if res.effect_size else "",
                "warnings_count": len(res.warnings),
            }
            rows.append(row)
        return pd.DataFrame(rows)


class AnalysisPipeline:
    """High-level orchestration service for SAES statistical workflow."""

    def __init__(self, alpha: float = 0.05, plotter: Any | None = None) -> None:
        self.alpha = alpha
        self.validator = DatasetValidator()
        self.decision_engine = TestDecisionEngine(alpha=alpha)
        self._plotter = plotter

    def load_csv(self, source: Any) -> Dataset:
        """Load CSV data source into a Dataset entity.

        Args:
            source: Path string or file buffer.

        Returns:
            Dataset object.
        """
        reader = CSVReader()
        df = reader.read(source)
        return Dataset(df=df)

    def validate(self, dataset: Dataset) -> ValidationReport:
        """Validate dataset schema and structure."""
        return self.validator.validate(dataset.df)

    def analyze(
        self,
        dataset: Dataset,
        generate_plots: bool = True,
        force_test: StatisticalTest | None = None,
    ) -> AnalysisReport:
        """Execute automated statistical analysis across all cells in dataset.

        Args:
            dataset: Validated SAES dataset.
            generate_plots: Whether to generate boxplots for each cell.
            force_test: Optional explicit StatisticalTest to override decision engine.

        Returns:
            AnalysisReport containing results for all (instance, metric) cells.
        """
        self.validator.validate_or_raise(dataset.df)

        report = AnalysisReport()
        grouped = dataset.df.groupby(["instance", "metric"])

        for (inst, met), cell_df in grouped:
            result = self.decision_engine.analyze_cell(cell_df, force_test=force_test)

            fig: Any | None = None
            if generate_plots:
                if self._plotter is None:
                    from saesV2.visualization.boxplot import BoxplotGenerator

                    self._plotter = BoxplotGenerator()
                fig = self._plotter.plot(cell_df, instance=str(inst), metric=str(met))

            report.cell_analyses.append(
                CellAnalysis(
                    instance=str(inst),
                    metric=str(met),
                    result=result,
                    figure=fig,
                )
            )

        return report
