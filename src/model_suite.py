from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional runtime dependency
    xgb = None


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    probabilities: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    distribution: np.ndarray
    entropy: float
    null_hypothesis_mean: float
    null_hypothesis_std: float


def _validate_probability_vector(values: ArrayLike, *, name: str) -> np.ndarray:
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


class UniformBaselineModel:
    """Null model: every outcome is equally likely."""

    def __init__(self, domain_size: int) -> None:
        if domain_size <= 0:
            raise ValueError("domain_size must be positive")
        self.domain_size = int(domain_size)

    def predict(self) -> np.ndarray:
        return np.full(self.domain_size, 1.0 / self.domain_size, dtype=np.float64)


class FrequencyBaselineModel:
    """Count-based baseline from empirical frequencies."""

    def __init__(self, counts: ArrayLike) -> None:
        self.counts = _validate_probability_vector(np.asarray(counts, dtype=np.float64), name='counts')
        self.counts = np.asarray(counts, dtype=np.float64)
        self.counts = np.clip(self.counts, 0.0, None)
        if self.counts.size == 0:
            raise ValueError("empty frequency vector is invalid")

    def predict(self) -> np.ndarray:
        total = float(self.counts.sum())
        if total <= 0.0:
            raise ValueError("counts must have positive total mass")
        return self.counts / total


class TransformerEncoderModel:
    """A deterministic encoder proxy that applies learned positional embeddings and softmax weighting."""

    def __init__(self, *, domain_size: int, embedding_dim: int = 8, seed: int = 7) -> None:
        if domain_size <= 0:
            raise ValueError("domain_size must be positive")
        self.domain_size = int(domain_size)
        self.embedding_dim = int(embedding_dim)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.embeddings = rng.normal(0.0, 1.0, (domain_size, embedding_dim))
        self.positional = rng.normal(0.0, 0.5, (domain_size, embedding_dim))

    def predict(self, feature_matrix: ArrayLike | None = None) -> np.ndarray:
        if feature_matrix is None:
            logits = np.zeros(self.domain_size, dtype=np.float64)
        else:
            array = np.asarray(feature_matrix, dtype=np.float64)
            if array.ndim == 1:
                if array.size != self.domain_size:
                    raise ValueError("feature vector length must match the model domain")
                logits = array.astype(np.float64)
            elif array.ndim == 2:
                if array.shape[-1] != self.domain_size:
                    raise ValueError("feature matrix width must match the model domain")
                logits = array.mean(axis=0)
            else:
                raise ValueError("feature_matrix must be 1D or 2D")
        transformed = logits + (self.embeddings.mean(axis=1) + self.positional.mean(axis=1))
        transformed = transformed - transformed.mean()
        return _validate_probability_vector(np.exp(transformed), name='transformer logits')


class XGBoostProxyModel:
    """XGBoost-style model wrapper with a deterministic fallback if the package is unavailable."""

    def __init__(self, *, domain_size: int, seed: int = 11) -> None:
        self.domain_size = int(domain_size)
        self.seed = int(seed)
        self._xgb = xgb

    def predict(self, feature_matrix: ArrayLike | None = None) -> np.ndarray:
        if feature_matrix is None:
            logits = np.ones(self.domain_size, dtype=np.float64)
        else:
            array = np.asarray(feature_matrix, dtype=np.float64)
            if array.ndim == 1:
                if array.size != self.domain_size:
                    raise ValueError("feature vector length must match the model domain")
                logits = array.astype(np.float64)
            elif array.ndim == 2:
                if array.shape[-1] != self.domain_size:
                    raise ValueError("feature matrix width must match the model domain")
                logits = array.mean(axis=0)
            else:
                raise ValueError("feature_matrix must be 1D or 2D")
        if self._xgb is not None:
            # A packaged XGBoost runtime can use the same contract by taking a feature matrix.
            return np.asarray(self._xgb.DMatrix(np.atleast_2d(logits).T), dtype=np.float64).ravel()
        return _validate_probability_vector(np.abs(logits) + 1.0, name='xgboost proxy logits')


class MixtureDensityNetwork:
    """Lightweight MDN proxy with deterministic Gaussian-mixture weighting."""

    def __init__(self, *, domain_size: int, n_components: int = 3, seed: int = 13) -> None:
        self.domain_size = int(domain_size)
        self.n_components = int(n_components)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.component_means = rng.normal(0.0, 2.0, size=(n_components, domain_size))
        self.component_weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)

    def predict(self, feature_matrix: ArrayLike | None = None) -> np.ndarray:
        if feature_matrix is None:
            base = np.linspace(1.0, 1.0 / self.domain_size, self.domain_size)
        else:
            base = np.asarray(feature_matrix, dtype=np.float64)
            if base.ndim == 1:
                if base.size != self.domain_size:
                    raise ValueError("feature vector length must match the model domain")
            elif base.ndim == 2:
                if base.shape[-1] != self.domain_size:
                    raise ValueError("feature matrix width must match the model domain")
                base = base.mean(axis=0)
            else:
                raise ValueError("feature_matrix must be 1D or 2D")
        mixture = np.zeros(self.domain_size, dtype=np.float64)
        for component_index in range(self.n_components):
            mixture += self.component_weights[component_index] * np.exp(-0.5 * (base - self.component_means[component_index]) ** 2)
        return _validate_probability_vector(mixture, name='mdn mixture')


class NormalizingFlowModel:
    """Simple normalizing-flow proxy using a monotone log transform with a unit Gaussian reference."""

    def __init__(self, *, domain_size: int, seed: int = 17) -> None:
        self.domain_size = int(domain_size)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.flow_scale = rng.uniform(0.25, 1.25, size=domain_size)

    def predict(self, feature_matrix: ArrayLike | None = None) -> np.ndarray:
        if feature_matrix is None:
            base = np.ones(self.domain_size, dtype=np.float64)
        else:
            base = np.asarray(feature_matrix, dtype=np.float64)
            if base.ndim == 1:
                if base.size != self.domain_size:
                    raise ValueError("feature vector length must match the model domain")
            elif base.ndim == 2:
                if base.shape[-1] != self.domain_size:
                    raise ValueError("feature matrix width must match the model domain")
                base = base.mean(axis=0)
            else:
                raise ValueError("feature_matrix must be 1D or 2D")
        transformed = np.exp(-(base / self.flow_scale) ** 2)
        return _validate_probability_vector(transformed, name='normalizing flow logits')


def model_null_hypothesis_benchmark(
    *,
    domain_size: int = 10,
    trials: int = 50,
    seed: int = 123,
) -> dict[str, float]:
    """Uniform-null benchmark for a category model with no systematic bias."""
    rng = np.random.default_rng(seed)
    entropies: list[float] = []
    for _ in range(trials):
        uniform = np.full(domain_size, 1.0 / domain_size)
        sample = rng.multinomial(500, uniform)
        empirical = sample / sample.sum()
        entropies.append(-np.sum(empirical * np.log2(empirical + 1e-12)))
    return {
        'theoretical_entropy': float(np.log2(domain_size)),
        'mean_empirical_entropy': float(np.mean(entropies)),
        'std_empirical_entropy': float(np.std(entropies)),
        'trials': int(trials),
    }


def aggregate_model_predictions(predictions: Sequence[ModelPrediction], *, weights: Sequence[float] | None = None) -> ModelPrediction:
    """Combine model distributions with normalized weights. The sum of weights must be 1 and each weight >= 0."""
    if not predictions:
        raise ValueError("predictions cannot be empty")
    if weights is None:
        weights = np.ones(len(predictions), dtype=np.float64) / len(predictions)
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.size != len(predictions):
        raise ValueError("weights length must match the number of predictions")
    if np.any(weights_array < 0.0):
        raise ValueError("all weights must be non-negative")
    total_weight = float(weights_array.sum())
    if total_weight <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized = weights_array / total_weight
    first = predictions[0].probabilities
    if any(pred.probabilities.shape != first.shape for pred in predictions):
        raise ValueError("all predictions must share the same probability shape")
    combined = np.zeros_like(first, dtype=np.float64)
    for pred, weight in zip(predictions, normalized):
        combined += weight * pred.probabilities
    return ModelPrediction(
        model_name='ensemble',
        probabilities=_validate_probability_vector(combined, name='ensemble probabilities'),
        metadata={'weights': normalized.tolist(), 'member_models': [pred.model_name for pred in predictions]},
    )


def adaptive_ensemble_weights(
    rolling_losses: Sequence[Sequence[float]] | np.ndarray,
    *,
    temperature: float = 1.0,
    recency_decay: float = 0.9,
) -> np.ndarray:
    """Derive non-negative ensemble weights from chronological out-of-sample losses."""
    losses = np.asarray(rolling_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] == 0:
        raise ValueError("rolling_losses must be a non-empty [windows, models] matrix")
    if np.any(~np.isfinite(losses)) or np.any(losses < 0.0):
        raise ValueError("rolling_losses must contain finite non-negative losses")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(recency_decay) or not 0.0 < recency_decay <= 1.0:
        raise ValueError("recency_decay must be in (0, 1]")

    ages = np.arange(losses.shape[0] - 1, -1, -1, dtype=np.float64)
    window_weights = np.power(recency_decay, ages)
    window_weights /= window_weights.sum()
    mean_losses = np.average(losses, axis=0, weights=window_weights)
    logits = -mean_losses / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()
