import math

import numpy as np

from src.feature_engineering import (
    cross_model_importance,
    feature_drift_report,
    monte_carlo_feature_drift_benchmark,
    stability_index,
)


def test_stability_index_constant_series_is_maximal() -> None:
    series = np.ones(20, dtype=np.float64)
    assert stability_index(series) == 1.0


def test_stability_index_zero_mean_series_is_rejected() -> None:
    series = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    assert stability_index(series) == 0.0


def test_cross_model_importance_averages_importances() -> None:
    importance_matrix = np.array([
        [0.1, 0.2, 0.3],
        [0.3, 0.4, 0.5],
    ])
    result = cross_model_importance(importance_matrix)
    expected = np.array([0.2, 0.3, 0.4])
    assert np.allclose(result, expected)


def test_feature_drift_report_tracks_window_to_window_change() -> None:
    series = np.linspace(0.0, 10.0, 40)
    report = feature_drift_report(series, feature_name='trend', window_size=10, step=5)
    assert len(report) == 1
    assert report[0].feature_name == 'trend'
    assert report[0].relative_drift >= 0.0
    assert math.isfinite(report[0].stability_index)


def test_monte_carlo_feature_drift_benchmark_stays_bounded() -> None:
    summary = monte_carlo_feature_drift_benchmark(n_features=25, n_samples=200, window_size=20)
    assert summary['mean_relative_drift'] >= 0.0
    assert summary['max_relative_drift'] >= summary['mean_relative_drift']
    assert summary['window_size'] == 20
