from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional runtime dependency
    xgb = None


# A type alias for any 1D probability-like vector.
ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class ModelPrediction:
    """
    Container for model predictions.

    Attributes:
        model_name: Name of the model.
        probabilities: Probability vector.
        metadata: Additional model-specific metadata.
    """
    model_name: str
    probabilities: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelResult:
    """
    Summary of model evaluation under a null hypothesis benchmark.

    Attributes:
        model_name: Name of the model.
        distribution: Probability distribution.
        entropy: Shannon entropy of the distribution.
        null_hypothesis_mean: Mean entropy under H0.
        null_hypothesis_std: Std entropy under H0.
    """
    model_name: str
    distribution: np.ndarray
    entropy: float
    null_hypothesis_mean: float
    null_hypothesis_std: float


def _validate_probability_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    """
    Validate and normalize a probability vector.

    Ensures:
        - 1D shape
        - non-negative values
        - positive finite mass
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
        # Validate counts as probabilities, then reapply raw counts with clipping.
        self.counts = _validate_probability_vector(np.asarray(counts, dtype=np.float64), name="counts")
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
    """
    Deterministic transformer-style encoder proxy.

    Applies:
        - learned embeddings
        - positional encodings
        - softmax normalization
    """

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

        return _validate_probability_vector(np.exp(transformed), name="transformer logits")


class XGBoostProxyModel:
    """XGBoost-style model wrapper with deterministic fallback."""

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
            # Deterministic fallback: convert logits to DMatrix and extract raw values.
            return np.asarray(self._xgb.DMatrix(np.atleast_2d(logits).T), dtype=np.float64).ravel()

        return _validate_probability_vector(np.abs(logits) + 1.0, name="xgboost proxy logits")


class MixtureDensityNetwork:
    """Lightweight MDN proxy using Gaussian mixture weighting."""

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

        for idx in range(self.n_components):
            mixture += self.component_weights[idx] * np.exp(
                -0.5 * (base - self.component_means[idx]) ** 2
            )

        return _validate_probability_vector(mixture, name="mdn mixture")


class NormalizingFlowModel:
    """Simple normalizing-flow proxy using monotone log transform."""

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
        return _validate_probability_vector(transformed, name="normalizing flow logits")


def model_null_hypothesis_benchmark(
        *,
        domain_size: int = 10,
        trials: int = 50,
        seed: int = 123,
) -> dict[str, float]:
    """
    Uniform-null benchmark for a category model with no systematic bias.

    Computes:
        - empirical entropy distribution
        - theoretical entropy log2(domain_size)
    """
    rng = np.random.default_rng(seed)
    entropies: list[float] = []

    for _ in range(trials):
        uniform = np.full(domain_size, 1.0 / domain_size)
        sample = rng.multinomial(500, uniform)
        empirical = sample / sample.sum()
        entropies.append(-np.sum(empirical * np.log2(empirical + 1e-12)))

    return {
        "theoretical_entropy": float(np.log2(domain_size)),
        "mean_empirical_entropy": float(np.mean(entropies)),
        "std_empirical_entropy": float(np.std(entropies)),
        "trials": int(trials),
    }


def aggregate_model_predictions(predictions: Sequence[ModelPrediction], *, weights: Sequence[float] | None = None) -> ModelPrediction:
    """
    Combine model probability distributions using normalized weights.

    Requirements:
        - all weights >= 0
        - sum(weights) > 0
        - all distributions share same shape
    """
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
        model_name="ensemble",
        probabilities=_validate_probability_vector(combined, name="ensemble probabilities"),
        metadata={
            "weights": normalized.tolist(),
            "member_models": [pred.model_name for pred in predictions],
        },
    )


def adaptive_ensemble_weights(
        rolling_losses: Sequence[Sequence[float]] | np.ndarray,
        *,
        temperature: float = 1.0,
        recency_decay: float = 0.9,
) -> np.ndarray:
    """
    Compute ensemble weights from chronological out-of-sample losses.

    Steps:
        - apply recency decay
        - compute weighted mean losses
        - convert losses to logits
        - softmax normalization
    """
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


# Función para aggarar predicciones de múltiples modelos
def aggregate_model_predictions(predictions: list, weights: list = None, method: str = "weighted_average") -> dict:
    """Aggregate model predictions using specified method.
    
    Parameters:
        predictions: list of prediction dicts with 'probability' key
        weights: list of weights (if None, use equal weights)
        method: 'weighted_average', 'median', 'max'
    
    Returns:
        dict with aggregated prediction
    """
    if not predictions:
        return {"probability": 0, "numbers": [], "confidence": 0}

    if method == "weighted_average":
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        weighted_sum = sum(p.probabilities.sum() * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        return ModelPrediction(model_name="ensemble", probabilities=np.array([weighted_sum / total_weight]),
                               metadata={})

    elif method == "median":
        probabilities = [p.probabilities.sum() for p in predictions]
        return ModelPrediction(model_name="ensemble",
                               probabilities=np.array([sorted(probabilities)[len(probabilities) // 2]]), metadata={})

    elif method == "max":
        best = max(predictions, key=lambda x: x.probabilities.sum())
        return best

    return predictions[0]


# Función para calcular accuracy de un modelo
def calculate_model_accuracy(y_true, y_pred) -> float:
    """Calculate accuracy percentage for model predictions."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return (len(y_true) / len(y_pred)) * 100 if len(y_pred) > 0 else 0.0
