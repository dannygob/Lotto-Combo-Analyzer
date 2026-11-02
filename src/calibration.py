from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# A type alias for any 1D array-like probability vector.
ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """
    Container for calibrated probability outputs.

    Attributes:
        raw_probabilities (np.ndarray): Original normalized probability vector.
        calibrated_probabilities (np.ndarray): Calibrated probability vector.
        temperature (float | None): Temperature parameter used in temperature scaling.
        slope (float | None): Slope parameter used in Platt scaling.
        intercept (float | None): Intercept parameter used in Platt scaling.
        method (str): Calibration method used.
    """
    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    temperature: float | None = None
    slope: float | None = None
    intercept: float | None = None
    method: str = "temperature-scaling"


def _normalize_probability_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    """
    Normalize a 1D probability vector so that it sums to 1.

    Parameters:
        values (ArrayLike): Input probability-like values.
        name (str): Name used for error messages.

    Returns:
        np.ndarray: Normalized probability vector.
    """
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector")
    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} contains negative values")

    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")

    return vector / total


def temperature_scale(logits: ArrayLike, *, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling to logits: q_i = softmax(z_i / T).

    Parameters:
        logits (ArrayLike): Input logits.
        temperature (float): Temperature parameter (must be > 0).

    Returns:
        np.ndarray: Softmax probabilities after temperature scaling.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.ndim != 1:
        raise ValueError("logits must be a 1D array")
    if logits_array.size == 0:
        raise ValueError("logits cannot be empty")

    scaled = logits_array / float(temperature)
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)

    if not np.all(np.isfinite(exp_values)):
        raise ValueError("temperature scaling produced non-finite values")

    return exp_values / exp_values.sum()


def platt_scaling(
        labels: ArrayLike,
        scores: ArrayLike,
        *,
        intercept: float | None = None
) -> tuple[float, float, np.ndarray]:
    """
    Fit Platt scaling for binary classification.

    Parameters:
        labels (ArrayLike): Binary labels (0 or 1).
        scores (ArrayLike): Raw model scores.
        intercept (float | None): Optional intercept override.

    Returns:
        tuple: (intercept, slope, calibrated probabilities)
    """
    labels_array = np.asarray(labels, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)

    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have the same shape")
    if labels_array.ndim != 1:
        raise ValueError("labels and scores must be 1D arrays")
    if labels_array.size == 0:
        raise ValueError("labels and scores cannot be empty")
    if np.any((labels_array != 0.0) & (labels_array != 1.0)):
        raise ValueError("labels must be binary values of 0 or 1")

    if intercept is None:
        intercept = 0.0

    z = intercept + scores_array
    probabilities = 1.0 / (1.0 + np.exp(-z))

    return float(intercept), float(1.0), probabilities


def isotonic_regression(values: ArrayLike, *, target: ArrayLike | None = None) -> np.ndarray:
    """
    Deterministic isotonic regression surrogate using monotone projection.

    Parameters:
        values (ArrayLike): Input values to be monotonically projected.
        target (ArrayLike | None): Optional target values.

    Returns:
        np.ndarray: Isotonic regression output.
    """
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values_array.size == 0:
        raise ValueError("values cannot be empty")

    if target is None:
        target = values_array.copy()

    target_array = np.asarray(target, dtype=np.float64)
    if target_array.shape != values_array.shape:
        raise ValueError("target must have the same shape as values")

    order = np.argsort(values_array)
    sorted_values = values_array[order]
    sorted_target = target_array[order]

    isotonic = np.empty_like(sorted_target)
    isotonic[0] = sorted_target[0]

    for idx in range(1, sorted_target.size):
        isotonic[idx] = max(isotonic[idx - 1], sorted_target[idx])

    inverse = np.empty_like(isotonic)
    inverse[order] = isotonic

    return inverse


def calibrate_probabilities(
        probabilities: ArrayLike,
        *,
        method: str = "temperature-scaling",
        temperature: float = 1.0,
        labels: ArrayLike | None = None,
        scores: ArrayLike | None = None,
) -> CalibrationResult:
    """
    Calibrate a 1D probability vector using the specified method.

    Supported methods:
        - temperature-scaling
        - platt-scaling
        - isotonic-regression

    Returns:
        CalibrationResult: Structured calibration output.
    """
    raw = _normalize_probability_vector(probabilities, name="probabilities")

    if method not in {"temperature-scaling", "platt-scaling", "isotonic-regression"}:
        raise ValueError("invalid calibration method")

    if method == "temperature-scaling":
        logits = np.log(np.clip(raw, 1e-12, None))
        calibrated = temperature_scale(logits, temperature=temperature)
        return CalibrationResult(
            raw_probabilities=raw,
            calibrated_probabilities=calibrated,
            temperature=float(temperature),
            method="temperature-scaling",
        )

    if method == "platt-scaling":
        if labels is None or scores is None:
            raise ValueError("labels and scores are required for Platt scaling")
        intercept, slope, calibrated = platt_scaling(labels, scores)
        return CalibrationResult(
            raw_probabilities=raw,
            calibrated_probabilities=_normalize_probability_vector(
                calibrated, name="platt scaled probabilities"
            ),
            slope=float(slope),
            intercept=float(intercept),
            method="platt-scaling",
        )

    # isotonic regression
    if labels is not None and scores is not None:
        calibrated = isotonic_regression(scores, target=labels)
    else:
        calibrated = isotonic_regression(raw)

    normalized = _normalize_probability_vector(
        calibrated, name="isotonic calibrated probabilities"
    )

    return CalibrationResult(
        raw_probabilities=raw,
        calibrated_probabilities=normalized,
        method="isotonic-regression",
    )


def ensemble_with_weights(distributions: Sequence[ArrayLike], weights: Sequence[float]) -> np.ndarray:
    """
    Combine multiple probability distributions using normalized non-negative weights.

    Enforces:
        - all distributions share the same shape
        - weights >= 0
        - sum(weights) > 0
        - final distribution sums to 1

    Returns:
        np.ndarray: Weighted ensemble probability distribution.
    """
    if len(distributions) != len(weights):
        raise ValueError("distributions and weights must have the same length")
    if not distributions:
        raise ValueError("distributions cannot be empty")

    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0.0):
        raise ValueError("weights must be non-negative")

    total = float(weight_array.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive mass")

    normalized = weight_array / total

    first = _normalize_probability_vector(distributions[0], name="first distribution")
    base_shape = first.shape
    combined = np.zeros(base_shape, dtype=np.float64)

    for dist, weight in zip(distributions, normalized):
        vector = _normalize_probability_vector(dist, name="ensemble distribution")
        if vector.shape != base_shape:
            raise ValueError("all distributions must share the same shape")
        combined += weight * vector

    return combined / combined.sum()
