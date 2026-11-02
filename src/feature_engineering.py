from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.information_theory import js_divergence


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class FeatureDrift:
    feature_name: str
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    relative_drift: float
    stability_index: float


@dataclass(frozen=True)
class FeatureWindow:
    start_index: int
    end_index: int
    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class DriftDiagnostic:
    divergence: float
    threshold: float
    drift_detected: bool
    baseline_count: int
    current_count: int


def _to_1d_numeric(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D numeric array")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def stability_index(values: ArrayLike) -> float:
    """Compute S(f) = 1 - sigma(f_t) / mu(f_t), with robust handling of zero-mean or constant series."""
    series = _to_1d_numeric(values, name='values')
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=0))
    if abs(mean) < 1e-12:
        return 1.0 if std <= 1e-12 else 0.0
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("stability_index requires finite values")
    return float(1.0 - (std / abs(mean)))


def cross_model_importance(importances: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the mean feature importance across models: I(f) = (1/M) * sum_m I_m(f)."""
    matrix = np.asarray(importances, dtype=np.float64)
    if matrix.size == 0:
        raise ValueError("importances must contain at least one model")
    if matrix.ndim != 2:
        raise ValueError("importances must be a 2D array-like structure: [model, feature]")
    if matrix.shape[1] == 0:
        raise ValueError("feature importance matrix must contain at least one feature")
    return matrix.mean(axis=0)


def chronological_feature_matrix(values: Sequence[ArrayLike], *, names: Sequence[str] | None = None) -> np.ndarray:
    """Stack time-indexed features into a 2D array shaped [time, features]."""
    if not values:
        raise ValueError("values must contain at least one feature series")
    series_list = [_to_1d_numeric(item, name=f'feature_{index}') for index, item in enumerate(values)]
    lengths = {len(item) for item in series_list}
    if len(lengths) != 1:
        raise ValueError("all feature series must share the same length")
    matrix = np.column_stack(series_list)
    if names is not None and len(names) != matrix.shape[1]:
        raise ValueError("names length must match the number of feature columns")
    return matrix


def rolling_windows(values: ArrayLike, *, window_size: int, step: int = 1) -> list[FeatureWindow]:
    """Compute rolling window statistics for a single series."""
    series = _to_1d_numeric(values, name='values')
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if window_size > len(series):
        raise ValueError("window_size cannot exceed the series length")
    windows: list[FeatureWindow] = []
    start = 0
    while start + window_size <= len(series):
        segment = series[start:start + window_size]
        windows.append(
            FeatureWindow(
                start_index=start,
                end_index=start + window_size - 1,
                mean=float(np.mean(segment)),
                std=float(np.std(segment, ddof=0)),
                count=int(segment.size),
            )
        )
        start += step
    if not windows:
        raise ValueError("windowing produced no segments")
    return windows


def feature_drift_report(
        values: ArrayLike,
        *,
        feature_name: str = 'feature',
        window_size: int = 10,
        step: int = 1,
) -> list[FeatureDrift]:
    """Track drift between the first and last rolling windows of a feature time series."""
    series = _to_1d_numeric(values, name=feature_name)
    windows = rolling_windows(series, window_size=window_size, step=step)
    if len(windows) < 2:
        raise ValueError("feature_drift_report requires at least two windows")
    baseline = windows[0]
    current = windows[-1]
    baseline_mean = baseline.mean
    current_mean = current.mean
    baseline_std = baseline.std
    current_std = current.std
    relative_drift = abs(current_mean - baseline_mean) / max(abs(baseline_mean), 1e-12)
    drift = FeatureDrift(
        feature_name=feature_name,
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        baseline_std=baseline_std,
        current_std=current_std,
        relative_drift=relative_drift,
        stability_index=stability_index(series),
    )
    return [drift]


def compare_feature_distributions(
        baseline_values: ArrayLike,
        current_values: ArrayLike,
        *,
        bins: int = 10,
        threshold: float = 0.05,
) -> DriftDiagnostic:
    """Compare two chronological feature samples using Jensen-Shannon divergence."""
    baseline = _to_1d_numeric(baseline_values, name="baseline_values")
    current = _to_1d_numeric(current_values, name="current_values")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    minimum = float(min(np.min(baseline), np.min(current)))
    maximum = float(max(np.max(baseline), np.max(current)))
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    edges = np.linspace(minimum, maximum, bins + 1)
    baseline_histogram, _ = np.histogram(baseline, bins=edges)
    current_histogram, _ = np.histogram(current, bins=edges)
    divergence = js_divergence(baseline_histogram, current_histogram)
    return DriftDiagnostic(
        divergence=divergence,
        threshold=float(threshold),
        drift_detected=divergence > threshold,
        baseline_count=int(baseline.size),
        current_count=int(current.size),
    )


def monte_carlo_feature_drift_benchmark(
        *,
        n_features: int = 10,
        n_samples: int = 250,
        seed: int = 42,
        window_size: int = 25,
) -> dict[str, float]:
    """Pure H0 benchmark: uniform random noise should not create persistent drift beyond a small tolerance."""
    rng = np.random.default_rng(seed)
    noisy = rng.standard_normal((n_samples, n_features))
    drifts = [
        feature_drift_report(noisy[:, idx], feature_name=f'feature_{idx}', window_size=window_size, step=1)[0].relative_drift
        for idx in range(n_features)
    ]
    mean_drift = float(np.mean(drifts))
    max_drift = float(np.max(drifts))
    return {
        'mean_relative_drift': mean_drift,
        'max_relative_drift': max_drift,
        'window_size': window_size,
        'n_features': n_features,
        'n_samples': n_samples,
    }
