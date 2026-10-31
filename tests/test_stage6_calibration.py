import math

import numpy as np

from src.calibration import (
    calibrate_probabilities,
    ensemble_with_weights,
    temperature_scale,
)


def test_temperature_scaling_preserves_probability_mass() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    probabilities = temperature_scale(logits, temperature=2.0)
    assert probabilities.shape == logits.shape
    assert math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert np.all(probabilities >= 0.0)


def test_temperature_scaling_reduces_extremity() -> None:
    logits = np.array([5.0, 1.0, 1.0], dtype=np.float64)
    scaled = temperature_scale(logits, temperature=10.0)
    assert scaled[0] < 0.8
    assert float(scaled.sum()) == 1.0


def test_calibrate_probabilities_temperature_scaling_is_valid() -> None:
    result = calibrate_probabilities(np.array([0.7, 0.2, 0.1]), temperature=1.5)
    assert result.method == 'temperature-scaling'
    assert result.temperature == 1.5
    assert math.isclose(float(result.calibrated_probabilities.sum()), 1.0, abs_tol=1e-12)


def test_isotonic_regression_keeps_non_negative_valid_probabilities() -> None:
    result = calibrate_probabilities(np.array([0.1, 0.2, 0.7]), method='isotonic-regression')
    assert result.method == 'isotonic-regression'
    assert np.all(result.calibrated_probabilities >= 0.0)
    assert math.isclose(float(result.calibrated_probabilities.sum()), 1.0, abs_tol=1e-12)


def test_ensemble_weights_are_normalized_and_non_negative() -> None:
    ensemble = ensemble_with_weights([
        np.array([0.9, 0.1]),
        np.array([0.2, 0.8]),
    ], [0.25, 0.75])
    assert math.isclose(float(ensemble.sum()), 1.0, abs_tol=1e-12)
    assert np.all(ensemble >= 0.0)
