from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.domain import NumberPool, Ruleset, SourceFile
from src.draw_schedule import (
    LotterySchedule,
    local_event_datetime,
    scheduled_session_datetime,
)
from src.information_theory import js_divergence
from src.preprocessing import infer_number_columns
from src.registry import SQLiteRegistry

MINIMUM_DATASET_ROWS = 20
MINIMUM_EVALUATION_ROWS = 1
CALIBRATION_ECE_THRESHOLD = 0.10
DRIFT_JS_THRESHOLD = 0.10
SMOOTHING_STRENGTH = 10.0
BOOTSTRAP_SAMPLES = 2000


# ============================================================================
# FUNCIONES DE MÉTRICAS DE CLASIFICACIÓN Y REGRESIÓN (UTILIZADAS)
# ============================================================================

def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """Calculate classification metrics from true labels and predictions."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss,
        confusion_matrix, matthews_corrcoef, cohen_kappa_score
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    
    auc_roc = None
    auc_pr = None
    log_loss_val = None
    
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
            log_loss_val = log_loss(y_true, y_proba)
        except ValueError:
            pass
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
        "auc_roc": float(auc_roc) if auc_roc is not None else None,
        "auc_pr": float(auc_pr) if auc_pr is not None else None,
        "log_loss": float(log_loss_val) if log_loss_val is not None else None,
    }


def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics from true and predicted values."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = 1
    r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "r2_adjusted": float(r2_adjusted),
    }


def calculate_residual_analysis(y_true, y_pred):
    """Calculate detailed residual analysis."""
    errors = y_pred - y_true
    
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    
    skewness = float(np.mean((errors - np.mean(errors))**3) / (np.std(errors)**3)) if np.std(errors) != 0 else 0.0
    kurtosis = float(np.mean((errors - np.mean(errors))**4) / (np.std(errors)**4) - 3) if np.std(errors) != 0 else 0.0
    
    return {
        "mean_error": mean_error,
        "std_error": std_error,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors)),
    }


# ============================================================================
# FUNCIONES DE ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

def calculate_histogram(data, bins=20):
    """Calculate histogram data."""
    counts, bin_edges = np.histogram(data, bins=bins)
    return {
        "bins": [float(x) for x in bin_edges],
        "counts": [int(x) for x in counts],
        "min_value": float(np.min(data)),
        "max_value": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }


def calculate_box_plot(data):
    """Calculate box plot statistics."""
    q1 = float(np.percentile(data, 25))
    median = float(np.percentile(data, 50))
    q3 = float(np.percentile(data, 75))
    iqr = q3 - q1
    lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
    upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
    outliers = [float(x) for x in data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]]
    
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "min": lower_whisker,
        "max": upper_whisker,
        "outliers": outliers,
        "iqr": iqr,
    }


def calculate_correlation_matrix(dataframe):
    """Calculate correlation matrix."""
    corr = dataframe.corr()
    return {
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


# ============================================================================
# FUNCIONES DE CLUSTERING Y REDUCCIÓN DE DIMENSIONALIDAD
# ============================================================================

def calculate_elbow_method(k_range, inertia):
    """Calculate elbow method data."""
    return {
        "k_range": k_range,
        "inertia": [float(x) for x in inertia],
        "optimal_k": int(np.argmin(np.gradient(inertia)) + 1) if len(inertia) > 1 else 1,
    }


def calculate_silhouette_plot(k, silhouette_avg, cluster_sizes):
    """Calculate silhouette plot data."""
    return {
        "k": k,
        "silhouette_avg": float(silhouette_avg),
        "cluster_sizes": cluster_sizes,
    }


def calculate_dendrogram(linkage_matrix, labels):
    """Calculate dendrogram data."""
    return {
        "linkage_matrix": linkage_matrix,
        "labels": labels,
    }


def calculate_dimensionality_reduction(method, points, explained_variance_ratio=None):
    """Calculate dimensionality reduction data."""
    return {
        "method": method,
        "points": points,
        "explained_variance_ratio": explained_variance_ratio or [],
    }


def _dataset_name(filename: str) -> str:
    normalized = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("uploaded CSV must have a valid filename")
    if not normalized.lower().endswith(".csv"):
        raise ValueError("uploaded file must have a .csv extension")
    return normalized


def _read_csv(raw_csv: bytes, filename: str) -> tuple[pd.DataFrame, int]:
    if not raw_csv:
        raise ValueError("uploaded CSV is empty")
    try:
        decoded = raw_csv.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ValueError("uploaded CSV must be a non-empty UTF-8 CSV file") from exc
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV header contains an empty column name")
    if len(header) != len(set(header)):
        raise ValueError("CSV column names must be unique")
    try:
        frame = pd.read_csv(io.BytesIO(raw_csv), encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"could not parse uploaded CSV {filename!r}: {exc}") from exc
    blank_rows = frame.isna().all(axis=1)
    blank_rows_skipped = int(blank_rows.sum())
    frame = frame.loc[~blank_rows].reset_index(drop=True)
    if frame.empty:
        raise ValueError("CSV must contain at least one draw")
    return frame, blank_rows_skipped


def inspect_csv_draw_format(raw_csv: bytes, filename: str) -> dict[str, object]:
    """Suggest a single-pool draw format from CSV columns and observed values."""
    safe_filename = _dataset_name(filename)
    frame, blank_rows_skipped = _read_csv(raw_csv, safe_filename)
    number_columns = infer_number_columns(frame.columns)
    suggestion: dict[str, int] | None = None
    warning = "Suggestions describe this file only; confirm the lottery's official rules."

    if number_columns:
        values = frame[number_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if np.all(np.isfinite(values)) and np.all(values == np.floor(values)):
            suggestion = {
                "minimum": int(values.min()),
                "maximum": int(values.max()),
                "numbers_drawn": len(number_columns),
            }
        else:
            warning = "Number-like columns contain missing or non-integer values; confirm the format manually."

    date_candidates = [
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in ("date", "fecha", "draw_date", "datetime"))
    ]
    event_id_candidates = [
        str(column)
        for column in frame.columns
        if str(column).strip().lower() in {
            "draw_id", "drawid", "event_id", "eventid", "sorteo",
            "draw_no", "draw_number", "numero_sorteo", "número de sorteo",
        }
    ]
    earliest_draw_date = None
    if date_candidates:
        parsed_dates = pd.to_datetime(frame[date_candidates[0]], errors="coerce")
        if parsed_dates.notna().all():
            earliest_draw_date = parsed_dates.min().date().isoformat()

    return {
        "filename": safe_filename,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "blank_rows_skipped": blank_rows_skipped,
        "number_columns": number_columns,
        "date_columns": date_candidates,
        "event_id_columns": event_id_candidates,
        "earliest_draw_date": earliest_draw_date,
        "suggested_format": suggestion,
        "warning": warning,
    }


def _resolve_pool_columns(
    frame: pd.DataFrame,
    ruleset_pools: Sequence[NumberPool],
    requested: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    pools_by_id = {pool.pool_id: pool for pool in ruleset_pools}
    if requested is None:
        if len(pools_by_id) != 1:
            raise ValueError("pool_columns mapping is required when the ruleset has multiple pools")
        requested = {next(iter(pools_by_id)): infer_number_columns(frame.columns)}
    if set(requested) != set(pools_by_id):
        raise ValueError("pool_columns keys must exactly match the registered ruleset pools")

    resolved: dict[str, list[str]] = {}
    seen_columns: set[str] = set()
    for pool_id, pool in pools_by_id.items():
        columns = list(requested[pool_id])
        if len(columns) != pool.numbers_drawn:
            raise ValueError(
                f"pool {pool_id!r} requires {pool.numbers_drawn} columns; received {len(columns)}"
            )
        if len(columns) != len(set(columns)):
            raise ValueError(f"pool {pool_id!r} contains duplicate column names")
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing columns for pool {pool_id!r}: {sorted(missing)}")
        if seen_columns.intersection(columns):
            raise ValueError("each CSV number column must belong to exactly one pool")
        seen_columns.update(columns)
        resolved[pool_id] = columns
    return resolved


def _parse_local_times(values: pd.Series, column_name: str) -> tuple[list[pd.Timedelta], list[str]]:
    durations: list[pd.Timedelta] = []
    normalized_times: list[str] = []
    for raw_value in values:
        value = str(raw_value).strip()
        parsed_time = None
        for time_format in ("%H:%M", "%H:%M:%S"):
            try:
                parsed_time = datetime.strptime(value, time_format).time()
                break
            except ValueError:
                continue
        if parsed_time is None:
            raise ValueError(
                f"draw time column {column_name!r} must contain local times in HH:MM or HH:MM:SS format"
            )
        durations.append(
            pd.Timedelta(
                hours=parsed_time.hour,
                minutes=parsed_time.minute,
                seconds=parsed_time.second,
            )
        )
        normalized_times.append(parsed_time.strftime("%H:%M:%S"))
    return durations, normalized_times


def preview_csv_schema(
    raw_csv: bytes,
    *,
    filename: str,
    lottery_id: str,
    ruleset: Ruleset,
) -> dict[str, object]:
    """Suggest column roles without training; ambiguous number-pool mappings require confirmation."""
    safe_filename = _dataset_name(filename)
    frame, blank_rows = _read_csv(raw_csv, safe_filename)
    number_candidates = infer_number_columns(frame.columns)
    pool_suggestions: dict[str, list[str]] = {}
    pools = list(ruleset.pools)

    if len(pools) == 1 and len(number_candidates) == pools[0].numbers_drawn:
        pool_suggestions[pools[0].pool_id] = number_candidates
    elif number_candidates and len(number_candidates) <= 16:
        eligible: dict[str, list[str]] = {}
        for pool in pools:
            eligible[pool.pool_id] = []
            for column in number_candidates:
                values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
                if (
                    values.size > 0
                    and np.all(np.isfinite(values))
                    and np.all(values == np.floor(values))
                    and np.all((values >= pool.minimum) & (values <= pool.maximum))
                ):
                    eligible[pool.pool_id].append(column)
        solutions: list[dict[str, list[str]]] = []

        def find_assignments(pool_index: int, used: set[str], assignment: dict[str, list[str]]) -> None:
            if len(solutions) > 1:
                return
            if pool_index == len(pools):
                solutions.append({key: value.copy() for key, value in assignment.items()})
                return
            pool = pools[pool_index]
            remaining = [column for column in eligible[pool.pool_id] if column not in used]
            for group in itertools.combinations(remaining, pool.numbers_drawn):
                assignment[pool.pool_id] = list(group)
                find_assignments(pool_index + 1, used.union(group), assignment)
                assignment.pop(pool.pool_id, None)

        find_assignments(0, set(), {})
        if len(solutions) == 1:
            pool_suggestions = solutions[0]

    date_candidates = [
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in ("date", "fecha", "draw_date", "datetime"))
    ]
    event_id_candidates = [
        str(column)
        for column in frame.columns
        if str(column).strip().lower() in {
            "draw_id", "drawid", "event_id", "eventid", "sorteo",
            "draw_no", "draw_number", "numero_sorteo", "número de sorteo",
        }
    ]
    session_candidates = [
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in ("session", "turno", "sesion", "draw_time"))
    ]
    classified_candidates = set(date_candidates + event_id_candidates + session_candidates)
    classified_candidates.update(
        column
        for columns in pool_suggestions.values()
        for column in columns
    )
    metadata_candidates = [
        str(column)
        for column in frame.columns
        if str(column) not in classified_candidates
    ]
    suggestions_unambiguous = (
        len(pool_suggestions) == len(pools)
        and all(len(pool_suggestions[pool.pool_id]) == pool.numbers_drawn for pool in pools)
    )
    return {
        "lottery_id": lottery_id,
        "ruleset_version": ruleset.version,
        "filename": safe_filename,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "number_column_candidates": number_candidates,
        "suggested_pool_columns": pool_suggestions,
        "mapping_unambiguous": suggestions_unambiguous,
        "confirmation_required": not suggestions_unambiguous,
        "date_column_candidates": date_candidates,
        "event_id_column_candidates": event_id_candidates,
        "session_column_candidates": session_candidates,
        "other_columns": metadata_candidates,
        "blank_rows_skipped": blank_rows,
        "notice": "Column roles are suggestions only; confirm the mapping before importing.",
    }


def _prepare_rows(
    frame: pd.DataFrame,
    *,
    ruleset: Ruleset,
    pool_columns: Mapping[str, Sequence[str]],
    date_column: str | None,
    event_id_column: str | None,
    session_column: str | None,
    draw_time_column: str | None,
    schedule: LotterySchedule | None,
    chronological_order_confirmed: bool,
) -> tuple[pd.DataFrame, list[str], int]:
    frame = frame.copy()
    number_columns = [column for columns in pool_columns.values() for column in columns]
    missing_draw_rows = frame[number_columns].isna().all(axis=1)
    rows_without_draw_numbers_skipped = int(missing_draw_rows.sum())
    frame = frame.loc[~missing_draw_rows].reset_index(drop=True)
    if frame.empty:
        raise ValueError("CSV contains no draw rows with number data")

    for metadata_column in (date_column, event_id_column, session_column, draw_time_column):
        if metadata_column is not None and metadata_column not in frame.columns:
            raise ValueError(f"event metadata column {metadata_column!r} is missing")

    if event_id_column is not None:
        event_ids = frame[event_id_column].astype("string").str.strip()
        if event_ids.isna().any() or event_ids.eq("").any():
            raise ValueError(f"event ID column {event_id_column!r} contains missing values")
        if event_ids.duplicated().any():
            raise ValueError(f"event ID column {event_id_column!r} must be unique")
        frame["_event_id"] = event_ids.astype(str)
    else:
        frame["_event_id"] = pd.Series([None] * len(frame), dtype="object")

    if session_column is not None:
        sessions = frame[session_column].astype("string").str.strip()
        if sessions.isna().any() or sessions.eq("").any():
            raise ValueError(f"session column {session_column!r} contains missing values")
        frame["_event_session"] = sessions.astype(str)
    else:
        frame["_event_session"] = pd.Series([None] * len(frame), dtype="object")

    parsed_dates: pd.Series | None = None
    if date_column is not None:
        parsed_dates = pd.to_datetime(frame[date_column], errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(f"date column {date_column!r} contains invalid values")
        frame[date_column] = parsed_dates.dt.strftime("%Y-%m-%d")
        for value in parsed_dates.dt.date:
            if value < ruleset.effective_from or (
                ruleset.effective_to is not None and value >= ruleset.effective_to
            ):
                raise ValueError("CSV contains dates outside the selected ruleset effective range")
    elif not chronological_order_confirmed:
        raise ValueError("provide date_column or confirm that CSV row order is chronological")

    for column in number_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"number column {column!r} contains missing or non-numeric values")
        if np.any(values != np.floor(values)):
            raise ValueError(f"number column {column!r} contains non-integer values")
        frame[column] = values.astype(np.int64)

    for pool in ruleset.pools:
        columns = pool_columns[pool.pool_id]
        values = frame[columns].to_numpy(dtype=np.int64)
        if np.any((values < pool.minimum) | (values > pool.maximum)):
            raise ValueError(
                f"pool {pool.pool_id!r} contains values outside [{pool.minimum}, {pool.maximum}]"
            )
        if not pool.allow_replacement and any(
            len(set(row.tolist())) != len(row) for row in values
        ):
            raise ValueError(f"pool {pool.pool_id!r} contains duplicate numbers in a draw")

    if parsed_dates is not None:
        date_keys = parsed_dates.dt.strftime("%Y-%m-%d")
        if (
            date_keys.duplicated().any()
            and event_id_column is None
            and session_column is None
            and draw_time_column is None
        ):
            raise ValueError(
                "duplicate draw dates require an event_id_column, session_column, or draw_time_column"
            )
        if date_keys.duplicated().any() and schedule is None and draw_time_column is None:
            raise ValueError(
                "same-day events require draw_time_column or a configured schedule to establish chronology"
            )
        identity_columns: dict[str, object] = {"event_date": date_keys}
        if session_column is not None:
            identity_columns["session"] = frame["_event_session"]
        if draw_time_column is not None:
            duration_values, local_times = _parse_local_times(
                frame[draw_time_column],
                draw_time_column,
            )
            durations = pd.Series(duration_values, index=frame.index)
            identity_columns["draw_time"] = local_times
        if (
            event_id_column is None
            and (session_column is not None or draw_time_column is not None)
            and pd.DataFrame(identity_columns).duplicated().any()
        ):
            raise ValueError("event date and session/time combination must be unique")

        if schedule is not None:
            event_datetimes: list[datetime] = []
            if session_column is not None:
                session_ids = frame["_event_session"].astype(str).tolist()
                for row_index, (event_date, session_id) in enumerate(
                    zip(parsed_dates.dt.date, session_ids, strict=True)
                ):
                    event_datetimes.append(
                        scheduled_session_datetime(
                            schedule,
                            event_date,
                            session_id,
                            local_time_override=(
                                local_times[row_index] if draw_time_column is not None else None
                            ),
                        )
                    )
            elif draw_time_column is not None:
                event_datetimes = [
                    local_event_datetime(
                        event_date,
                        local_times[row_index],
                        timezone_name=schedule.timezone,
                    )
                    for row_index, event_date in enumerate(parsed_dates.dt.date)
                ]
            elif len(schedule.sessions) == 1:
                session_id = schedule.sessions[0].session_id
                event_datetimes = [
                    scheduled_session_datetime(schedule, event_date, session_id)
                    for event_date in parsed_dates.dt.date
                ]
            else:
                raise ValueError(
                    "session_column is required to order draws when the configured schedule has multiple sessions"
                )
            frame["_event_timestamp"] = pd.to_datetime(
                [event_datetime.astimezone(timezone.utc) for event_datetime in event_datetimes],
                utc=True,
            )
        elif draw_time_column is not None:
            frame["_event_timestamp"] = parsed_dates.dt.normalize() + durations
        else:
            frame["_event_timestamp"] = parsed_dates.dt.normalize()
        if date_keys.duplicated().any() and frame["_event_timestamp"].duplicated().any():
            raise ValueError(
                "same-day events must have distinct scheduled timestamps for chronological ordering"
            )
        sort_columns = ["_event_timestamp"]
        if event_id_column is not None:
            tie_breaker = frame["_event_id"]
        elif session_column is not None:
            tie_breaker = frame["_event_session"]
        elif draw_time_column is not None:
            tie_breaker = local_times
        else:
            tie_breaker = frame["_event_id"]
        if tie_breaker is not None:
            tie_breaker_series = pd.Series(tie_breaker, index=frame.index)
            numeric_tie_breaker = pd.to_numeric(tie_breaker_series, errors="coerce")
            frame["_event_tie_breaker"] = (
                numeric_tie_breaker
                if numeric_tie_breaker.notna().all()
                else tie_breaker_series.astype(str)
            )
            sort_columns.append("_event_tie_breaker")
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    elif event_id_column is not None:
        frame = frame.sort_values("_event_id", kind="stable").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    if event_id_column is None:
        if date_column is not None and session_column is not None:
            frame["_event_id"] = (
                frame[date_column].astype(str)
                + ":"
                + frame["_event_session"].astype(str)
                + (
                    ":" + frame[draw_time_column].astype(str)
                    if draw_time_column is not None
                    else ""
                )
            )
        elif date_column is not None and draw_time_column is not None:
            frame["_event_id"] = (
                frame[date_column].astype(str) + ":" + frame[draw_time_column].astype(str)
            )
        elif date_column is not None:
            frame["_event_id"] = frame[date_column].astype(str)
        else:
            frame["_event_id"] = [f"row-{index + 1}" for index in range(len(frame))]
    return frame, number_columns, rows_without_draw_numbers_skipped


def _expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(outcomes[mask]))
            )
    return ece


def _train_and_evaluate(
    frame: pd.DataFrame,
    *,
    ruleset: Ruleset,
    pool_columns: Mapping[str, Sequence[str]],
) -> tuple[dict[str, object], dict[str, object]]:
    row_count = len(frame)
    if row_count < 3:
        raise ValueError("at least 3 chronological draws are required for leakage-free training and evaluation")
    first_test_index = max(2, row_count // 2)
    if first_test_index >= row_count:
        first_test_index = row_count - MINIMUM_EVALUATION_ROWS

    per_draw_model_brier: list[np.ndarray] = []
    per_draw_uniform_brier: list[np.ndarray] = []
    flattened_probabilities: list[np.ndarray] = []
    flattened_outcomes: list[np.ndarray] = []
    temporal_brier_by_feature: dict[str, list[np.ndarray]] = {}
    pool_models: dict[str, dict[str, object]] = {}
    pool_drift: dict[str, float] = {}
    temporal_contexts: dict[str, np.ndarray] = {}
    if "_event_timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["_event_timestamp"])
        temporal_contexts["weekday"] = timestamps.dt.day_of_week.astype(str).to_numpy()
        temporal_contexts["month"] = timestamps.dt.month.astype(str).to_numpy()
        temporal_contexts["quarter_of_year"] = timestamps.dt.quarter.astype(str).to_numpy()
        gaps = timestamps.diff().dt.total_seconds().div(86400).fillna(0.0).to_numpy()
        temporal_contexts["inter_draw_gap"] = np.select(
            [gaps == 0, gaps <= 1, gaps <= 3, gaps <= 7],
            ["same-day", "1-day", "2-3-days", "4-7-days"],
            default="8+-days",
        )

    for pool in ruleset.pools:
        columns = list(pool_columns[pool.pool_id])
        domain = np.arange(pool.minimum, pool.maximum + 1, dtype=np.int64)
        domain_size = len(domain)
        actual = np.zeros((row_count, domain_size), dtype=np.float64)
        for row_index, row in enumerate(frame[columns].to_numpy(dtype=np.int64)):
            actual[row_index, row - pool.minimum] = 1.0

        model_predictions: list[np.ndarray] = []
        temporal_predictions: dict[str, list[np.ndarray]] = {
            feature: [] for feature in temporal_contexts
        }
        seen_counts = actual[:first_test_index].sum(axis=0)
        for test_index in range(first_test_index, row_count):
            seen_count = test_index
            prior_inclusion = pool.numbers_drawn / domain_size
            prediction = (
                seen_counts + SMOOTHING_STRENGTH * prior_inclusion
            ) / (seen_count + SMOOTHING_STRENGTH)
            model_predictions.append(prediction)
            for feature_name, feature_values in temporal_contexts.items():
                matching_rows = feature_values[:test_index] == feature_values[test_index]
                matching_counts = actual[:test_index][matching_rows].sum(axis=0)
                matching_count = int(matching_rows.sum())
                temporal_predictions[feature_name].append(
                    (matching_counts + SMOOTHING_STRENGTH * prediction)
                    / (matching_count + SMOOTHING_STRENGTH)
                )
            seen_counts = seen_counts + actual[test_index]

        predictions = np.asarray(model_predictions, dtype=np.float64)
        observed = actual[first_test_index:]
        uniform = np.full(
            (len(observed), domain_size),
            pool.numbers_drawn / domain_size,
            dtype=np.float64,
        )
        model_brier = np.mean(np.square(predictions - observed), axis=1)
        uniform_brier = np.mean(np.square(uniform - observed), axis=1)
        per_draw_model_brier.append(model_brier)
        per_draw_uniform_brier.append(uniform_brier)
        for feature_name, feature_probabilities in temporal_predictions.items():
            feature_matrix = np.asarray(feature_probabilities, dtype=np.float64)
            feature_brier = np.mean(np.square(feature_matrix - observed), axis=1)
            temporal_brier_by_feature.setdefault(feature_name, []).append(feature_brier)
        flattened_probabilities.append(predictions.ravel())
        flattened_outcomes.append(observed.ravel())

        training_counts = actual.sum(axis=0)
        full_probabilities = (
            training_counts + SMOOTHING_STRENGTH * (pool.numbers_drawn / domain_size)
        ) / (row_count + SMOOTHING_STRENGTH)
        if not math.isclose(float(full_probabilities.sum()), pool.numbers_drawn, rel_tol=1e-10):
            raise ArithmeticError(f"trained inclusion probabilities for pool {pool.pool_id!r} are not normalized")

        recent_start = max(first_test_index, row_count // 2)
        early_distribution = actual[:first_test_index].sum(axis=0)
        recent_distribution = actual[recent_start:].sum(axis=0)
        if early_distribution.sum() <= 0 or recent_distribution.sum() <= 0:
            pool_drift[pool.pool_id] = 0.0
        else:
            pool_drift[pool.pool_id] = js_divergence(early_distribution, recent_distribution)
        pool_models[pool.pool_id] = {
            "minimum": pool.minimum,
            "maximum": pool.maximum,
            "numbers_drawn": pool.numbers_drawn,
            "number_columns": columns,
            "probabilities": full_probabilities.tolist(),
            "counts": training_counts.astype(int).tolist(),
            "training_rows": row_count,
            "smoothing_strength": SMOOTHING_STRENGTH,
        }

    model_brier = np.mean(np.vstack(per_draw_model_brier), axis=0)
    uniform_brier = np.mean(np.vstack(per_draw_uniform_brier), axis=0)
    differences = model_brier - uniform_brier
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(frame.to_csv(index=False).encode()).digest()[:8], "big"))
    bootstrap_means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for bootstrap_index in range(BOOTSTRAP_SAMPLES):
        bootstrap_means[bootstrap_index] = float(
            np.mean(rng.choice(differences, size=len(differences), replace=True))
        )
    confidence_interval = np.quantile(bootstrap_means, [0.025, 0.975])

    temporal_tests: dict[str, dict[str, object]] = {}
    unadjusted_p_values: dict[str, float] = {}
    for feature_name, pool_brier_values in temporal_brier_by_feature.items():
        temporal_brier = np.mean(np.vstack(pool_brier_values), axis=0)
        temporal_difference = temporal_brier - model_brier
        observed_difference = float(np.mean(temporal_difference))
        centered_difference = temporal_difference - observed_difference
        temporal_bootstrap = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
        temporal_effect_bootstrap = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
        for bootstrap_index in range(BOOTSTRAP_SAMPLES):
            temporal_bootstrap[bootstrap_index] = float(
                np.mean(
                    rng.choice(
                        centered_difference,
                        size=len(centered_difference),
                        replace=True,
                    )
                )
            )
            temporal_effect_bootstrap[bootstrap_index] = float(
                np.mean(
                    rng.choice(
                        temporal_difference,
                        size=len(temporal_difference),
                        replace=True,
                    )
                )
            )
        interval = np.quantile(temporal_effect_bootstrap, [0.025, 0.975])
        p_value = float(
            (1 + np.count_nonzero(temporal_bootstrap <= observed_difference))
            / (BOOTSTRAP_SAMPLES + 1)
        )
        unadjusted_p_values[feature_name] = p_value
        temporal_tests[feature_name] = {
            "brier_score": float(np.mean(temporal_brier)),
            "baseline_brier_score": float(np.mean(model_brier)),
            "mean_brier_difference_vs_baseline": float(np.mean(temporal_difference)),
            "brier_difference_95ci": [float(interval[0]), float(interval[1])],
            "bootstrap_p_value": p_value,
            "bootstrap_test": "one-sided null-centered paired bootstrap",
        }
    holm_order = sorted(unadjusted_p_values, key=unadjusted_p_values.get)
    running_adjusted_p = 0.0
    for rank, feature_name in enumerate(holm_order):
        adjusted = min(1.0, (len(holm_order) - rank) * unadjusted_p_values[feature_name])
        running_adjusted_p = max(running_adjusted_p, adjusted)
        temporal_tests[feature_name]["holm_adjusted_p_value"] = running_adjusted_p
        ci = temporal_tests[feature_name]["brier_difference_95ci"]
        temporal_tests[feature_name]["retained"] = bool(
            ci[1] < 0.0 and running_adjusted_p < 0.05
        )

    all_probabilities = np.concatenate(flattened_probabilities)
    all_outcomes = np.concatenate(flattened_outcomes)
    calibration_error = _expected_calibration_error(all_probabilities, all_outcomes)
    maximum_drift = max(pool_drift.values(), default=0.0)
    # --- MÉTRICAS EXTENDIDAS ---
    import math
    epsilon = 1e-15

    # 1. MAE (Error Absoluto Medio)
    mae = float(np.mean(np.abs(all_probabilities - all_outcomes))) if len(all_probabilities) > 0 else 0.0

    # 2. MSE (Error Cuadrático Medio)
    mse = float(np.mean((all_probabilities - all_outcomes) ** 2)) if len(all_probabilities) > 0 else 0.0

    # 3. RMSE (Raíz del MSE)
    rmse = float(math.sqrt(mse)) if mse >= 0 else 0.0

    # 4. Log Loss / Entropía Cruzada Binaria
    all_probabilities_clipped = np.clip(all_probabilities, epsilon, 1 - epsilon)
    log_loss = float(-np.mean(all_outcomes * np.log(all_probabilities_clipped) + (1 - all_outcomes) * np.log(1 - all_probabilities_clipped))) if len(all_probabilities) > 0 else 0.0

    # 5. Brier Score
    brier_score = float(np.mean((all_probabilities - all_outcomes) ** 2)) if len(all_probabilities) > 0 else 0.0

    # 6. sMAPE (Mean Absolute Percentage Error simétrico)
    smape = float(np.mean(2 * np.abs(all_probabilities - all_outcomes) / (np.abs(all_probabilities) + np.abs(all_outcomes) + epsilon))) if len(all_probabilities) > 0 else 0.0

    # 7. Forecast Bias (sesgo sistemático)
    forecast_bias = float(np.mean(all_probabilities - all_outcomes)) if len(all_probabilities) > 0 else 0.0

    # 8. F1 Score, Precision, Recall (clasificación binaria simplificada)
    y_true_binary = (all_outcomes > 0).astype(int)
    y_pred_binary = (all_probabilities > 0.5).astype(int)
    precision = float(np.sum((y_true_binary == 1) & (y_pred_binary == 1)) / np.sum(y_pred_binary == 1)) if np.sum(y_pred_binary == 1) > 0 else 0.0
    recall = float(np.sum((y_true_binary == 1) & (y_pred_binary == 1)) / np.sum(y_true_binary == 1)) if np.sum(y_true_binary == 1) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 9. Entropía Cruzada (equivalente a log_loss)
    cross_entropy = log_loss

    # 10. Divergencia KL (Kullback-Leibler) aproximada
    kl_divergence = float(np.mean(all_outcomes * np.log(all_outcomes / all_probabilities_clipped) + (1 - all_outcomes) * np.log((1 - all_outcomes) / (1 - all_probabilities_clipped)))) if len(all_probabilities) > 0 else 0.0

    # --- FIN MÉTRICAS EXTENDIDAS ---

    # Diccionario de métricas (se mantiene compatibilidad hacia abajo con las gates existentes)
    metrics: dict[str, object] = {
        # Métricas existentes del sistema (obligatorias para gates)
        "validation_passed": row_count >= MINIMUM_DATASET_ROWS,
        "leakage_free": first_test_index < row_count,
        "calibration_passed": calibration_error <= CALIBRATION_ECE_THRESHOLD,
        "beats_null": bool(confidence_interval[1] < 0.0),
        "drift_within_tolerance": maximum_drift <= DRIFT_JS_THRESHOLD,
        
        # --- NUEVAS MÉTRICAS COMPATIBLES ---
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "log_loss": log_loss,
        "brier_score": brier_score,
        "smape": smape,
        "cross_entropy": cross_entropy,
        "kl_divergence": kl_divergence,
        "forecast_bias": forecast_bias,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        
        # Metadatos y conteos
        "dataset_rows": row_count,
        "training_rows_initial": first_test_index,
        
        # --- MÉTRICAS POR POOL ---
        "pool_metrics": pool_metrics if 'pool_metrics' in locals() else {},
        "pool_frequency": pool_frequency if 'pool_frequency' in locals() else {},
    }

    
    artifact: dict[str, object] = {
        "model_type": "smoothed-frequency-inclusion",
        "model_version": None,
        "lottery_ruleset_version": ruleset.version,
        "dataset_rows": row_count,
        "training_start_index": 0,
        "training_end_index_exclusive": row_count,
        "pool_models": pool_models,
        "evaluation": metrics,
        "scientific_note": "Empirical frequencies are not evidence of future lottery predictability.",
    }
    return artifact, metrics


def calculate_pool_metrics(all_probabilities_by_pool, all_outcomes_by_pool, ruleset):
    """Calculate metrics for each pool separately."""
    pool_metrics = {}
    
    for pool in ruleset.pools:
        pool_id = pool.pool_id
        probabilities = all_probabilities_by_pool.get(pool_id, [])
        outcomes = all_outcomes_by_pool.get(pool_id, [])
        
        if len(probabilities) == 0 or len(outcomes) == 0:
            pool_metrics[pool_id] = {
                "mae": 0.0, "mse": 0.0, "rmse": 0.0,
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "log_loss": 0.0, "brier_score": 0.0,
                "true_positives": 0, "false_positives": 0,
                "false_negatives": 0, "true_negatives": 0,
                "accuracy": 0.0,
            }
            continue
        
        epsilon = 1e-15
        probabilities_clipped = np.clip(probabilities, epsilon, 1 - epsilon)
        
        y_true_binary = (outcomes > 0).astype(int)
        y_pred_binary = (probabilities > 0.5).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        mae = float(np.mean(np.abs(probabilities - outcomes)))
        mse = float(np.mean((probabilities - outcomes) ** 2))
        rmse = float(np.sqrt(mse))
        
        log_loss = float(-np.mean(outcomes * np.log(probabilities_clipped) + (1 - outcomes) * np.log(1 - probabilities_clipped)))
        brier_score = float(np.mean((probabilities - outcomes) ** 2))
        
        pool_metrics[pool_id] = {
            "mae": mae, "mse": mse, "rmse": rmse,
            "precision": float(precision), "recall": float(recall), "f1_score": float(f1_score),
            "log_loss": log_loss, "brier_score": brier_score,
            "true_positives": int(tp), "false_positives": int(fp),
            "false_negatives": int(fn), "true_negatives": int(tn),
            "accuracy": float(accuracy),
        }
    
    return pool_metrics


def calculate_pool_frequency_data(frame, pool_columns, ruleset):
    """Calculate frequency data per pool."""
    pool_frequency = {}
    
    for pool in ruleset.pools:
        pool_id = pool.pool_id
        columns = pool_columns.get(pool_id, [])
        
        if not columns:
            continue
        
        all_numbers = []
        for col in columns:
            col_data = frame[col].dropna().tolist()
            all_numbers.extend(col_data)
        
        from collections import Counter
        freq_counter = Counter(all_numbers)
        total_count = sum(freq_counter.values())
        
        frequencies = {}
        for num in range(pool.minimum, pool.maximum + 1):
            count = freq_counter.get(num, 0)
            frequencies[str(num)] = {
                "count": count,
                "frequency": count / total_count if total_count > 0 else 0.0,
            }
        
        pool_frequency[pool_id] = {
            "pool_name": pool_id,
            "numbers_drawn": pool.numbers_drawn,
            "minimum": pool.minimum,
            "maximum": pool.maximum,
            "total_numbers": len(all_numbers),
            "unique_numbers": len(freq_counter),
            "frequencies": frequencies,
        }
    
    return pool_frequency


def calculate_sum_distribution(pool_id: str, minimum: int, maximum: int, numbers_drawn: int) -> dict:
    """Calculate expected sum distribution for a pool."""
    min_sum = sum(range(1, numbers_drawn + 1))
    max_sum = sum(range(maximum - numbers_drawn + 1, maximum + 1))
    expected_sum = (minimum + maximum) * numbers_drawn / 2
    
    variance = (numbers_drawn * (maximum - minimum + 1)**2 * (maximum - minimum + 1 - numbers_drawn)) / ((maximum - minimum + 1)**2 * (maximum - minimum + 1 - 1))
    std_dev = variance ** 0.5
    
    return {
        "pool_id": pool_id,
        "minimum_sum": min_sum,
        "maximum_sum": max_sum,
        "expected_sum": expected_sum,
        "std_dev": std_dev,
        "one_std_range": [expected_sum - std_dev, expected_sum + std_dev],
        "two_std_range": [expected_sum - 2*std_dev, expected_sum + 2*std_dev],
        "is_within_normal_range": lambda s: (expected_sum - 2*std_dev) <= s <= (expected_sum + 2*std_dev),
    }


def filter_combinations_by_sum(combinations: list, pool_id: str, sum_distribution: dict) -> dict:
    """Filter combinations based on sum distribution."""
    filtered = {
        "accepted": [],
        "rejected": [],
        "rejection_reasons": {},
    }
    
    min_s = sum_distribution["minimum_sum"]
    max_s = sum_distribution["maximum_sum"]
    expected = sum_distribution["expected_sum"]
    std = sum_distribution["std_dev"]
    
    for combo in combinations:
        combo_sum = sum(combo)
        combo_id = f"{pool_id}_{'_'.join(map(str, combo))}"
        
        if combo_sum < min_s or combo_sum > max_s:
            filtered["rejected"].append(combo)
            filtered["rejection_reasons"][combo_id] = f"Suma {combo_sum} fuera de rango [{min_s}-{max_s}]"
            continue
        
        if not sum_distribution["is_within_normal_range"](combo_sum):
            filtered["rejected"].append(combo)
            filtered["rejection_reasons"][combo_id] = f"Suma {combo_sum} fuera de 2σ ({expected:.1f}±{2*std:.1f})"
            continue
        
        filtered["accepted"].append({
            "combination": combo,
            "sum": combo_sum,
            "deviation_from_expected": abs(combo_sum - expected),
            "z_score": abs(combo_sum - expected) / std if std > 0 else 0,
        })
    
    filtered["accepted"].sort(key=lambda x: x["deviation_from_expected"])
    
    return filtered


def adjust_pool_weights_by_performance(pool_metrics: dict) -> dict:
    """Adjust pool weights based on performance (F1 score and precision)."""
    pool_weights = {}
    total_score = 0.0
    
    for pool_id, metrics in pool_metrics.items():
        f1 = metrics.get("f1_score", 0.0)
        precision = metrics.get("precision", 0.0)
        accuracy = metrics.get("accuracy", 0.0)
        
        score = (f1 * 0.5) + (precision * 0.3) + (accuracy * 0.2)
        score = max(score, 0.1)
        
        pool_weights[pool_id] = score
        total_score += score
    
    if total_score > 0:
        for pool_id in pool_weights:
            pool_weights[pool_id] /= total_score
    else:
        num_pools = len(pool_metrics)
        for pool_id in pool_weights:
            pool_weights[pool_id] = 1.0 / num_pools
    
    return pool_weights


def adjust_prediction_threshold_by_calibration(calibration_passed: bool, drift_within_tolerance: bool, beats_null: bool) -> float:
    """Adjust prediction threshold based on model quality."""
    base_threshold = 0.5
    
    if not calibration_passed:
        base_threshold += 0.15
    if not drift_within_tolerance:
        base_threshold += 0.10
    if not beats_null:
        base_threshold += 0.20
    
    return max(0.3, min(0.95, base_threshold))


def apply_model_improvements(metrics: dict, pool_metrics: dict, predictions: dict, ruleset_pools: list) -> dict:
    """Apply all improvements to the model based on metrics."""
    
    pool_weights = adjust_pool_weights_by_performance(pool_metrics)
    
    calibration_passed = metrics.get("calibration_passed", False)
    drift_within_tolerance = metrics.get("drift_within_tolerance", True)
    beats_null = metrics.get("beats_null", False)
    
    threshold = adjust_prediction_threshold_by_calibration(calibration_passed, drift_within_tolerance, beats_null)
    
    sum_distributions = {}
    for pool in ruleset_pools:
        sum_distributions[pool.pool_id] = calculate_sum_distribution(pool.pool_id, pool.minimum, pool.maximum, pool.numbers_drawn)
    
    adjusted_predictions = {}
    
    for pool_id, pool_preds in predictions.items():
        weight = pool_weights.get(pool_id, 1.0 / len(pool_metrics)) if pool_metrics else 1.0
        
        adjusted_pool_preds = []
        for pred in pool_preds:
            weighted_prob = pred.get("probability", 0.0) * weight
            binary_pred = 1 if weighted_prob >= threshold else 0
            confidence = abs(weighted_prob - threshold)
            
            adjusted_pool_preds.append({
                "number": pred.get("number"),
                "original_probability": pred.get("probability", 0.0),
                "weighted_probability": weighted_prob,
                "binary_prediction": binary_pred,
                "confidence": confidence,
                "pool_weight": weight,
                "threshold_used": threshold,
            })
        
        adjusted_pool_preds.sort(key=lambda x: x["weighted_probability"], reverse=True)
        
        pool_info = next((p for p in ruleset_pools if p.pool_id == pool_id), None)
        if pool_info:
            adjusted_pool_preds = adjusted_pool_preds[:pool_info.numbers_drawn]
        
        adjusted_predictions[pool_id] = adjusted_pool_preds
    
    sum_validations = {}
    for pool_id, pool_preds in adjusted_predictions.items():
        combo = [p["number"] for p in pool_preds if p["binary_prediction"] == 1]
        if len(combo) >= 2:
            combo_sum = sum(combo)
            sum_dist = sum_distributions.get(pool_id, {})
            
            sum_validations[pool_id] = {
                "combination": combo,
                "sum": combo_sum,
                "expected_sum": sum_dist.get("expected_sum", 0),
                "std_dev": sum_dist.get("std_dev", 1),
                "z_score": abs(combo_sum - sum_dist.get("expected_sum", 0)) / sum_dist.get("std_dev", 1) if sum_dist.get("std_dev", 1) > 0 else 0,
                "is_normal": sum_dist.get("is_within_normal_range", lambda s: True)(combo_sum),
            }
    
    return {
        "adjusted_predictions": adjusted_predictions,
        "pool_weights": pool_weights,
        "prediction_threshold": threshold,
        "sum_distributions": sum_distributions,
        "sum_validations": sum_validations,
        "calibration_applied": not calibration_passed,
        "overfitting_detected": not drift_within_tolerance,
    }


def import_and_retrain_csv(
    registry: SQLiteRegistry,
    *,
    raw_csv: bytes,
    filename: str,
    lottery_id: str,
    ruleset_version: str,
    data_directory: str | Path,
    model_directory: str | Path,
    pool_columns: Mapping[str, Sequence[str]] | None = None,
    date_column: str | None = None,
    event_id_column: str | None = None,
    session_column: str | None = None,
    draw_time_column: str | None = None,
    chronological_order_confirmed: bool | None = None,
) -> dict[str, object]:
    """Create an immutable CSV snapshot and train a leakage-free candidate model."""
    safe_filename = _dataset_name(filename)
    ruleset = registry.get_ruleset(lottery_id, ruleset_version)
    if ruleset is None:
        raise ValueError("lottery ruleset must be registered before importing a CSV")
    schema_profile = registry.get_csv_schema_profile(lottery_id, ruleset_version)
    if schema_profile is not None:
        configured_columns = {
            "date_column": schema_profile.date_column,
            "event_id_column": schema_profile.event_id_column,
            "session_column": schema_profile.session_column,
            "draw_time_column": schema_profile.draw_time_column,
        }
        supplied_columns = {
            "date_column": date_column,
            "event_id_column": event_id_column,
            "session_column": session_column,
            "draw_time_column": draw_time_column,
        }
        for field_name, supplied_value in supplied_columns.items():
            configured_value = configured_columns[field_name]
            if supplied_value is not None and supplied_value != configured_value:
                raise ValueError(
                    f"{field_name} differs from the saved CSV schema profile; update the profile first"
                )
        if pool_columns is not None and {
            key: list(columns) for key, columns in pool_columns.items()
        } != schema_profile.pool_columns:
            raise ValueError("pool_columns differs from the saved CSV schema profile; update the profile first")
        if (
            chronological_order_confirmed is not None
            and chronological_order_confirmed != schema_profile.chronological_order_confirmed
        ):
            raise ValueError(
                "chronological_order_confirmed differs from the saved CSV schema profile; update the profile first"
            )
        pool_columns = schema_profile.pool_columns
        date_column = schema_profile.date_column
        event_id_column = schema_profile.event_id_column
        session_column = schema_profile.session_column
        draw_time_column = schema_profile.draw_time_column
        if chronological_order_confirmed is None:
            chronological_order_confirmed = schema_profile.chronological_order_confirmed
    if chronological_order_confirmed is None:
        chronological_order_confirmed = False
    frame, blank_rows_skipped = _read_csv(raw_csv, safe_filename)
    if pool_columns is None:
        preview = preview_csv_schema(
            raw_csv,
            filename=safe_filename,
            lottery_id=lottery_id,
            ruleset=ruleset,
        )
        if preview["confirmation_required"]:
            raise ValueError(
                "number column mapping is ambiguous; confirm pool_columns using the schema preview"
            )
        pool_columns = preview["suggested_pool_columns"]
    resolved_columns = _resolve_pool_columns(frame, ruleset.pools, pool_columns)
    if schema_profile is not None:
        expected_columns = {
            column
            for pool in schema_profile.pool_columns.values()
            for column in pool
        }
        expected_columns.update(
            column
            for column in (
                schema_profile.date_column,
                schema_profile.event_id_column,
                schema_profile.session_column,
                schema_profile.draw_time_column,
            )
            if column is not None
        )
        expected_columns.update(schema_profile.allowed_attribute_columns)
        expected_columns.update(schema_profile.ignored_columns)
        actual_columns = {str(column) for column in frame.columns}
        missing_profile_columns = expected_columns - actual_columns
        unclassified_columns = actual_columns - expected_columns
        if missing_profile_columns or unclassified_columns:
            details = []
            if missing_profile_columns:
                details.append(f"missing configured columns: {sorted(missing_profile_columns)}")
            if unclassified_columns:
                details.append(f"unclassified CSV columns: {sorted(unclassified_columns)}")
            raise ValueError("CSV does not match the saved schema profile; " + "; ".join(details))

    schedule = registry.get_lottery_schedule(lottery_id)
    frame, used_number_columns, rows_without_draw_numbers_skipped = _prepare_rows(
        frame,
        ruleset=ruleset,
        pool_columns=resolved_columns,
        date_column=date_column,
        event_id_column=event_id_column,
        session_column=session_column,
        draw_time_column=draw_time_column,
        schedule=schedule,
        chronological_order_confirmed=chronological_order_confirmed,
    )
    source_sha256 = hashlib.sha256(raw_csv).hexdigest()
    dataset_key = safe_filename
    csv_schema = {
        "lottery_id": lottery_id,
        "ruleset_version": ruleset_version,
        "pool_columns": resolved_columns,
        "date_column": date_column,
        "event_id_column": event_id_column,
        "session_column": session_column,
        "draw_time_column": draw_time_column,
        "allowed_attribute_columns": (
            schema_profile.allowed_attribute_columns if schema_profile is not None else []
        ),
        "ignored_columns": schema_profile.ignored_columns if schema_profile is not None else [],
        "chronological_order_confirmed": chronological_order_confirmed,
    }
    training_configuration = {
        "csv_schema": csv_schema,
        "schedule": schedule.model_dump(mode="json") if schedule is not None else None,
    }
    configuration_hash = hashlib.sha256(
        json.dumps(
            training_configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    model_seed = f"{lottery_id}:{dataset_key}:{source_sha256}:{configuration_hash}".encode()
    model_version = f"csv-{hashlib.sha256(model_seed).hexdigest()[:20]}"
    previous_active = next(
        (
            str(item["model_version"])
            for item in registry.list_model_versions(lottery_id)
            if item["status"] == "active"
        ),
        None,
    )
    prior_dataset_versions = [
        item
        for item in registry.list_dataset_versions(lottery_id)
        if item["dataset_key"] == dataset_key
    ]
    existing = registry.get_dataset_version(
        lottery_id=lottery_id,
        dataset_key=dataset_key,
        source_sha256=source_sha256,
    )
    if existing is not None:
        known_models = registry.list_model_versions(lottery_id)
        existing_model = next(
            (item for item in known_models if item["model_version"] == model_version),
            None,
        )
        if existing_model is not None:
            return {
                "status": "unchanged",
                "retrained": False,
                "dataset_id": existing["dataset_id"],
                "dataset_version": existing["version_number"],
                "dataset_sha256": source_sha256,
                "model_version": model_version,
                "model_status": existing_model["status"],
                "first_dataset": False,
                "previous_active_model": previous_active,
                "message": "This exact CSV snapshot has already been trained.",
            }

    artifact, metrics = _train_and_evaluate(
        frame,
        ruleset=ruleset,
        pool_columns=resolved_columns,
    )
    metrics["blank_rows_skipped"] = blank_rows_skipped
    metrics["rows_without_draw_numbers_skipped"] = rows_without_draw_numbers_skipped
    artifact["blank_rows_skipped"] = blank_rows_skipped
    artifact["rows_without_draw_numbers_skipped"] = rows_without_draw_numbers_skipped
    artifact["model_version"] = model_version
    artifact["csv_schema"] = csv_schema
    artifact["lottery_schedule"] = training_configuration["schedule"]

    dataset_root = Path(data_directory).resolve()
    model_root = Path(model_directory).resolve()
    snapshot_path = dataset_root / lottery_id / f"{source_sha256}.csv"
    artifact_path = model_root / lottery_id / f"{model_version}.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        snapshot_path.write_bytes(raw_csv)
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    registry.register_source_file_if_missing(
        SourceFile(
            sha256=source_sha256,
            original_name=safe_filename,
            storage_uri=str(snapshot_path),
            media_type="text/csv",
            size_bytes=len(raw_csv),
        )
    )
    dataset = existing or registry.register_dataset_version(
        lottery_id=lottery_id,
        dataset_key=dataset_key,
        source_sha256=source_sha256,
        row_count=len(frame),
        columns=list(frame.columns),
    )
    registry.register_model_version(
        model_version=model_version,
        lottery_id=lottery_id,
        dataset_sha256=source_sha256,
        artifact_uri=str(artifact_path),
        metrics=metrics,
    )
    required_gates = (
        "validation_passed",
        "leakage_free",
        "calibration_passed",
        "beats_null",
        "drift_within_tolerance",
    )
    promoted = all(metrics.get(gate) is True for gate in required_gates)
    if promoted:
        registry.promote_model_version(model_version)
    model_status = "active" if promoted else "candidate"
    registry.record_training_run(
        run_id=f"train-{hashlib.sha256(model_version.encode()).hexdigest()[:20]}",
        dataset_id=int(dataset["dataset_id"]),
        model_version=model_version,
        status=model_status,
        metrics=metrics,
    )
    return {
        "status": "trained",
        "retrained": True,
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version_number"],
        "first_dataset": len(prior_dataset_versions) == 0,
        "dataset_key": dataset_key,
        "dataset_sha256": source_sha256,
        "row_count": len(frame),
        "blank_rows_skipped": blank_rows_skipped,
        "rows_without_draw_numbers_skipped": rows_without_draw_numbers_skipped,
        "number_columns": used_number_columns,
        "model_version": model_version,
        "model_status": model_status,
        "previous_active_model": previous_active,
        "training_run_id": f"train-{hashlib.sha256(model_version.encode()).hexdigest()[:20]}",
        "promoted": promoted,
        "metrics": metrics,
        "snapshot_uri": str(snapshot_path),
        "artifact_uri": str(artifact_path),
    }
