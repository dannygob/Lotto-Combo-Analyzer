from __future__ import annotations

from typing import Sequence

import numpy as np


def probabilistic_diagnostics(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    outcomes: Sequence[int] | np.ndarray,
    *,
    regimes: Sequence[str] | None = None,
    calibration_bins: int = 10,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Measure out-of-sample categorical calibration and paired uniform-null log-loss lift."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(outcomes)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty [observations, classes] matrix")
    if labels.ndim != 1 or labels.size != matrix.shape[0]:
        raise ValueError("outcomes must contain one class index per probability row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("outcomes must contain integer class indices")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if np.any(labels < 0) or np.any(labels >= matrix.shape[1]):
        raise ValueError("outcome class index is outside the probability domain")
    if calibration_bins <= 0 or bootstrap_samples <= 0:
        raise ValueError("calibration_bins and bootstrap_samples must be positive")
    if regimes is not None and len(regimes) != labels.size:
        raise ValueError("regimes must contain one label per observation")

    totals = matrix.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("each probability row must have positive finite mass")
    normalized = matrix / totals[:, np.newaxis]
    rows = np.arange(labels.size)
    confidence = normalized.max(axis=1)
    predicted = normalized.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    selected_probabilities = np.clip(normalized[rows, labels], 1e-15, 1.0)
    per_observation_log_loss = -np.log(selected_probabilities)
    uniform_log_loss = float(np.log(matrix.shape[1]))
    loss_difference = per_observation_log_loss - uniform_log_loss

    bin_edges = np.linspace(0.0, 1.0, calibration_bins + 1)

    def expected_calibration_error(mask: np.ndarray) -> float:
        ece = 0.0
        group_size = int(mask.sum())
        for bin_index in range(calibration_bins):
            in_bin = (confidence >= bin_edges[bin_index]) & (
                confidence <= bin_edges[bin_index + 1]
                if bin_index == calibration_bins - 1
                else confidence < bin_edges[bin_index + 1]
            )
            in_bin &= mask
            if np.any(in_bin):
                ece += float(in_bin.sum() / group_size) * abs(
                    float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin]))
                )
        return ece

    rng = np.random.default_rng(seed)

    def summarize(mask: np.ndarray) -> dict[str, float | int | bool | list[float]]:
        count = int(mask.sum())
        if count == 0:
            raise ValueError("regime groups cannot be empty")
        observed_losses = per_observation_log_loss[mask]
        differences = loss_difference[mask]
        bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            sample = rng.choice(differences, size=count, replace=True)
            bootstrap_means[index] = float(np.mean(sample))
        interval = np.quantile(bootstrap_means, [0.025, 0.975])
        return {
            "observations": count,
            "brier_score": float(np.mean(np.sum((normalized[mask] - np.eye(matrix.shape[1])[labels[mask]]) ** 2, axis=1))),
            "log_loss": float(np.mean(observed_losses)),
            "uniform_log_loss": uniform_log_loss,
            "expected_calibration_error": expected_calibration_error(mask),
            "mean_log_loss_difference_vs_uniform": float(np.mean(differences)),
            "log_loss_difference_95ci": [float(interval[0]), float(interval[1])],
            "beats_uniform_at_95pct": bool(interval[1] < 0.0),
        }

    all_mask = np.ones(labels.size, dtype=bool)
    result: dict[str, object] = summarize(all_mask)
    if regimes is not None:
        regime_array = np.asarray(regimes, dtype=str)
        result["by_regime"] = {
            regime: summarize(regime_array == regime)
            for regime in sorted(set(regime_array.tolist()))
        }
    return result
