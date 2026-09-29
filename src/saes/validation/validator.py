"""Dataset schema and structural validation engine for SAES."""

import pandas as pd

from saes.domain.exceptions import ValidationError
from saes.domain.models import ValidationReport

REQUIRED_COLUMNS = ["algorithm", "instance", "metric", "run_id", "value", "maximize"]


class DatasetValidator:
    """Validates raw DataFrame structure, schema, column data types, missing values, and statistical requirements."""

    def __init__(self, min_sample_size: int = 2) -> None:
        self.min_sample_size = min_sample_size

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Perform comprehensive validation on a raw DataFrame.

        Args:
            df: Raw pandas DataFrame to validate.

        Returns:
            ValidationReport containing is_valid status and detailed list of issues.
        """
        report = ValidationReport(is_valid=True)

        # 1. Missing Required Columns check
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            report.add_error(
                issue_type="MissingColumns",
                message=f"Missing required column(s): {', '.join(missing_cols)}",
            )
            return report  # Cannot perform subsequent checks without required columns

        # 2. Check empty DataFrame
        if len(df) == 0:
            report.add_error(issue_type="EmptyDataset", message="Dataset contains 0 rows.")
            return report

        # 3. Check null / missing values
        null_counts = df[REQUIRED_COLUMNS].isnull().sum()
        for col in REQUIRED_COLUMNS:
            if null_counts[col] > 0:
                report.add_error(
                    issue_type="MissingValues",
                    message=f"Column '{col}' contains {null_counts[col]} null or missing value(s).",
                    column=col,
                )

        # 4. Column Type & Value Validation
        # algorithm, instance, metric: string validation
        for col in ["algorithm", "instance", "metric"]:
            non_str = df[df[col].notnull() & ~df[col].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0)]
            if not non_str.empty:
                for idx in non_str.index[:5]:  # limit sample reported rows
                    report.add_error(
                        issue_type="InvalidType",
                        message=f"Column '{col}' must contain non-empty string values. Found: '{df.loc[idx, col]}'",
                        column=col,
                        row_index=int(idx),
                    )

        # run_id: non-negative integer
        for idx, val in df["run_id"].dropna().items():
            try:
                val_int = int(val)
                if val_int != val or val_int < 0:
                    report.add_error(
                        issue_type="InvalidValue",
                        message=f"Column 'run_id' must be a non-negative integer. Found: {val}",
                        column="run_id",
                        row_index=int(str(idx)),
                    )
            except (ValueError, TypeError):
                report.add_error(
                    issue_type="InvalidType",
                    message=f"Column 'run_id' must be a non-negative integer. Found: {val}",
                    column="run_id",
                    row_index=int(str(idx)),
                )

        # value: numeric float/int
        for idx, val in df["value"].dropna().items():
            try:
                float(val)
            except (ValueError, TypeError):
                report.add_error(
                    issue_type="InvalidType",
                    message=f"Column 'value' must be numeric. Found: '{val}'",
                    column="value",
                    row_index=int(str(idx)),
                )

        # maximize: boolean
        bool_values = {True, False, "true", "false", "True", "False"}
        for idx, val in df["maximize"].dropna().items():
            if val not in bool_values and not isinstance(val, (bool, int)):
                report.add_error(
                    issue_type="InvalidType",
                    message=f"Column 'maximize' must be boolean (True/False). Found: '{val}'",
                    column="maximize",
                    row_index=int(str(idx)),
                )

        # 5. Check for duplicate observations (algorithm, instance, metric, run_id)
        if report.is_valid:
            duplicates = df.duplicated(subset=["algorithm", "instance", "metric", "run_id"], keep=False)
            if duplicates.any():
                dup_count = duplicates.sum()
                report.add_error(
                    issue_type="DuplicateObservation",
                    message=(
                        f"Found {dup_count} duplicate observation(s) with identical "
                        "(algorithm, instance, metric, run_id)."
                    ),
                )

        # 6. Statistical group adequacy check per (instance, metric) cell
        if report.is_valid:
            grouped = df.groupby(["instance", "metric"])
            for (inst, met), group in grouped:
                alg_counts = group.groupby("algorithm")["value"].count()
                if len(alg_counts) < 2:
                    report.add_warning(
                        issue_type="InsufficientGroups",
                        message=(
                            f"Experimental cell (instance='{inst}', metric='{met}') has only {len(alg_counts)} "
                            "algorithm group. Comparative statistical tests require >= 2 algorithms."
                        ),
                    )
                for alg, count in alg_counts.items():
                    if count < self.min_sample_size:
                        report.add_error(
                            issue_type="InsufficientSampleSize",
                            message=(
                                f"Algorithm '{alg}' in (instance='{inst}', metric='{met}') has only {count} "
                                f"observation(s). Minimum sample size is {self.min_sample_size}."
                            ),
                        )

        return report

    def validate_or_raise(self, df: pd.DataFrame) -> None:
        """Validate DataFrame and raise ValidationError if invalid.

        Raises:
            ValidationError: If dataset validation fails.
        """
        report = self.validate(df)
        if not report.is_valid:
            error_msgs = [str(err) for err in report.errors]
            raise ValidationError(
                f"Dataset validation failed with {len(report.errors)} error(s):\n" + "\n".join(error_msgs)
            )
