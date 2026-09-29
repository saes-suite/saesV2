"""Context generator for formatting statistical reports and datasets into prompts."""

from saes.app.pipeline import AnalysisReport
from saes.domain.models import Dataset


def build_analysis_context(report: AnalysisReport | None, dataset: Dataset | None = None) -> str:
    """Format analysis report and dataset summaries into context for the LLM.

    Args:
        report: AnalysisReport instance containing test results.
        dataset: Optional Dataset instance.

    Returns:
        Structured markdown text summarizing the experimental findings.
    """
    sections: list[str] = []

    if dataset is not None and not dataset.df.empty:
        df = dataset.df
        algorithms = sorted([str(a) for a in df["algorithm"].unique()])
        instances = sorted([str(i) for i in df["instance"].unique()])
        metrics = sorted([str(m) for m in df["metric"].unique()])

        sections.append("### Experimental Design Overview")
        sections.append(f"- **Algorithms evaluated ({len(algorithms)})**: {', '.join(algorithms)}")
        sections.append(f"- **Problem instances ({len(instances)})**: {', '.join(instances)}")
        sections.append(f"- **Performance metrics ({len(metrics)})**: {', '.join(metrics)}")
        sections.append(f"- **Total observations**: {len(df)} rows")

    if report is not None and report.cell_analyses:
        sections.append("\n### Statistical Test Results by Experimental Cell")
        for cell in report.cell_analyses:
            res = cell.result
            sig_label = "Significant" if res.is_significant else "Not significant"
            sections.append(f"#### Instance: `{cell.instance}` | Metric: `{cell.metric}`")
            sections.append(f"- **Applied Test**: {res.test_name}")
            sections.append(f"- **Statistic**: {res.statistic:.4f}")
            sections.append(f"- **p-value**: {res.p_value:.4e} ({sig_label} at alpha={res.alpha})")
            sections.append(f"- **Degrees of Freedom**: {res.degrees_of_freedom}")

            if res.effect_size:
                eff_text = f"{res.effect_size.name} = {res.effect_size.value:.4f} ({res.effect_size.magnitude})"
                sections.append(f"- **Effect Size**: {eff_text}")

            if res.assumptions:
                assumptions_summary: list[str] = []
                for a in res.assumptions:
                    status = "Passed" if a.passed else "Violated"
                    assumptions_summary.append(f"{a.name} (p={a.p_value:.3e}, {status})")
                sections.append(f"- **Assumptions**: {'; '.join(assumptions_summary)}")

            if res.warnings:
                sections.append(f"- **Warnings/Notes**: {'; '.join(res.warnings)}")

    if not sections:
        return "No statistical analysis or dataset context is currently loaded."

    return "\n".join(sections)
