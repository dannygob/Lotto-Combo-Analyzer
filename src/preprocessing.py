# preprocessing.py
# Data loading, cleaning, normalization, and ruleset validation utilities.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.domain import NumberPool, Ruleset


def load_dataset(path):
    """
    Load the lottery dataset from a CSV file.

    Parameters:
        path (str | Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(path)


def extract_combo_columns(df):
    """
    Extract columns that represent lottery numbers.

    Parameters:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        list[str]: Columns starting with 'numero '.
    """
    return [col for col in df.columns if col.startswith("numero ")]


def clean_numeric_columns(df, combo_cols):
    """
    Convert combo columns to numeric and drop rows with missing values.

    Parameters:
        df (pd.DataFrame): Raw DataFrame.
        combo_cols (list[str]): Columns containing lottery numbers.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.dropna(subset=combo_cols).copy()

    for col in combo_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=combo_cols).reset_index(drop=True)
    return df


def sort_combinations(df, combo_cols):
    """
    Sort each combination of numbers in ascending order.

    Parameters:
        df (pd.DataFrame): DataFrame with combo columns.
        combo_cols (list[str]): Columns containing lottery numbers.

    Returns:
        pd.Series: Sorted tuples per row.
    """
    return df[combo_cols].apply(lambda row: tuple(sorted(row)), axis=1)


def prepare_history_set(df_sorted):
    """
    Create a set of historical combinations for uniqueness checks.

    Parameters:
        df_sorted (pd.Series): Sorted combinations.

    Returns:
        set[tuple]: Unique combinations.
    """
    return set(df_sorted)


def infer_number_columns(columns: Iterable[str]) -> list[str]:
    """
    Infer number columns from typical lottery naming conventions.

    Recognizes:
        - 'numero x'
        - 'n{x}' where x is a digit sequence

    Returns:
        list[str]: Sorted list of inferred number columns.
    """
    normalized: list[str] = []

    for column in columns:
        name = str(column).strip()

        if name.lower().startswith("numero "):
            normalized.append(name)
        elif name.lower().startswith("n") and name[1:].isdigit():
            normalized.append(name)

    return sorted(normalized, key=lambda item: (item.lower(), item))


def normalize_date_column(df: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    """
    Normalize a date-like column into ISO format (YYYY-MM-DD).

    If date_column is None, the DataFrame is returned unchanged.
    """
    if date_column is None:
        return df

    if date_column not in df.columns:
        raise ValueError(f"date column {date_column!r} is missing")

    series = pd.to_datetime(df[date_column], errors="coerce")

    if series.isna().any():
        raise ValueError(f"date column {date_column!r} contains invalid values")

    df = df.copy()
    df[date_column] = series.dt.strftime("%Y-%m-%d")
    return df


def validate_number_frame(
        df: pd.DataFrame,
        number_columns: Sequence[str] | None = None,
        *,
        minimum: int = 1,
        maximum: int = 99,
        expected_count: int | None = None,
) -> pd.DataFrame:
    """
    Validate numeric columns for lottery draws.

    Ensures:
        - columns exist
        - numeric conversion succeeds
        - values fall within [minimum, maximum]
        - row completeness
        - expected count matches actual count (if provided)
    """
    if df.empty:
        raise ValueError("input frame is empty")

    if number_columns is None:
        number_columns = infer_number_columns(df.columns)

    if not number_columns:
        raise ValueError("no number columns were found to validate")

    if expected_count is not None and len(number_columns) != expected_count:
        raise ValueError(
            f"expected {expected_count} number columns but found {len(number_columns)}: {number_columns}"
        )

    cleaned = df.copy()

    # Convert to numeric
    for column in number_columns:
        if column not in cleaned.columns:
            raise ValueError(f"number column {column!r} is missing")
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=list(number_columns)).reset_index(drop=True)

    # Range validation
    for column in number_columns:
        out_of_range = ((cleaned[column] < minimum) | (cleaned[column] > maximum)).sum()
        if out_of_range:
            raise ValueError(f"column {column!r} contains values outside [{minimum}, {maximum}]")

    # Convert to integers
    for column in number_columns:
        cleaned[column] = cleaned[column].astype(int)

    return cleaned


def detect_ruleset_change(
        ruleset: Ruleset,
        frame: pd.DataFrame,
        number_columns: Sequence[str] | None = None,
) -> dict | None:
    """
    Detect whether a candidate draw frame suggests a ruleset transition.

    Checks:
        - column set matches pool IDs
        - draw count matches ruleset
        - min/max bounds match ruleset
    """
    if number_columns is None:
        number_columns = infer_number_columns(frame.columns)

    if not number_columns:
        return None

    pool_map = {pool.pool_id: pool for pool in ruleset.pools}

    # Column mismatch detection
    if len(number_columns) > 1 and set(number_columns) != set(pool_map):
        if set(number_columns) - set(pool_map):
            return {
                "reason": "number column set does not match the registered pool IDs",
                "observed_columns": list(number_columns),
                "expected_pools": sorted(pool_map),
            }

    numeric_frame = frame[list(number_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric_frame.to_numpy().ravel()
    valid_values = values[~pd.isna(values)]

    if valid_values.size == 0:
        return {"reason": "no valid numbers were found in the candidate frame"}

    changes = []

    for pool_id, pool in pool_map.items():
        if len(number_columns) != pool.numbers_drawn:
            changes.append({
                "pool_id": pool_id,
                "reason": "draw count differs",
                "expected_count": pool.numbers_drawn,
                "actual_count": len(number_columns),
            })

        if valid_values.min() < pool.minimum:
            changes.append({
                "pool_id": pool_id,
                "reason": "minimum bound differs",
                "expected_min": pool.minimum,
                "actual_min": int(valid_values.min()),
            })

        if valid_values.max() > pool.maximum:
            changes.append({
                "pool_id": pool_id,
                "reason": "maximum bound differs",
                "expected_max": pool.maximum,
                "actual_max": int(valid_values.max()),
            })

    return None if not changes else {"changes": changes}


def load_and_validate_history(
        path: str | Path,
        *,
        number_columns: Sequence[str] | None = None,
        date_column: str | None = None,
        minimum: int = 1,
        maximum: int = 99,
        expected_count: int | None = None,
) -> pd.DataFrame:
    """
    Load a lottery CSV and validate its structure for the scientific pipeline.

    Steps:
        - load dataset
        - infer number columns if needed
        - normalize date column if provided
        - validate numeric frame
    """
    frame = load_dataset(path)

    if number_columns is None:
        number_columns = infer_number_columns(frame.columns)

    if not number_columns:
        raise ValueError(f"no number columns were found in dataset {path!s}")

    if date_column is not None:
        frame = normalize_date_column(frame, date_column)

    return validate_number_frame(
        frame,
        number_columns=number_columns,
        minimum=minimum,
        maximum=maximum,
        expected_count=expected_count,
    )
