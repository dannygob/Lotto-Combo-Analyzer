from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# A type alias for any 1D probability-like vector.
ArrayLike = Sequence[float] | np.ndarray


def _as_probability_vector(values: ArrayLike, *, dimension: int | None = None) -> np.ndarray:
    """
    Convert input to a validated probability vector.

    Ensures:
        - 1D shape
        - non-negative values
        - positive finite mass
        - optional dimension check
    """
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("probability values must be passed as a 1D array-like object")

    if dimension is not None and array.size != dimension:
        raise ValueError(f"expected a vector of length {dimension}, received {array.size}")

    if array.size == 0:
        raise ValueError("probability vector cannot be empty")

    if np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative")

    total = float(array.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probability vector must have a positive finite mass")

    normalized = array / total

    if not np.all(np.isfinite(normalized)):
        raise ValueError("probability vector contains non-finite values")

    return normalized


def entropy(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    """
    Shannon entropy H(P) = -Σ p log_b p.

    Parameters:
        probabilities: Probability vector.
        base: Logarithm base.

    Returns:
        float: Shannon entropy.
    """
    if base <= 0.0:
        raise ValueError("base must be positive")

    p = _as_probability_vector(probabilities)
    mask = p > 0.0
    value = -(p[mask] * np.log(p[mask]) / np.log(base))
    return float(value.sum())


def conditional_entropy(joint_distribution: ArrayLike, *, base: float = 2.0) -> float:
    """
    Conditional entropy H(X|Y) from joint distribution P(X,Y).

    Parameters:
        joint_distribution: 2D matrix of joint probabilities.
        base: Logarithm base.

    Returns:
        float: Conditional entropy.
    """
    matrix = np.asarray(joint_distribution, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("joint_distribution must be a 2D array")

    if matrix.size == 0:
        raise ValueError("joint_distribution cannot be empty")

    if np.any(matrix < 0.0):
        raise ValueError("joint_distribution entries must be non-negative")

    total = float(matrix.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("joint_distribution must have positive finite mass")

    matrix = matrix / total
    p_y = matrix.sum(axis=0)

    total_entropy = 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p_xy = matrix[i, j]
            if p_xy <= 0.0:
                continue

            p_yj = p_y[j]
            if p_yj <= 0.0:
                raise ValueError("conditional entropy requires strictly positive marginal P(Y)")

            total_entropy -= p_xy * math.log(p_xy / p_yj, base)

    return float(total_entropy)


def mutual_information(joint_distribution: ArrayLike, *, base: float = 2.0) -> float:
    """
    Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

    Computed via:
        I(X;Y) = Σ P(x,y) log( P(x,y) / (P(x)P(y)) )
    """
    matrix = np.asarray(joint_distribution, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("joint_distribution must be a 2D array")

    if np.any(matrix < 0.0):
        raise ValueError("joint_distribution entries must be non-negative")

    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("joint_distribution must have positive mass")

    matrix = matrix / total

    p_x = matrix.sum(axis=1)
    p_y = matrix.sum(axis=0)

    mask = matrix > 0.0
    value = np.zeros_like(matrix, dtype=np.float64)

    denominators = p_x[:, None] * p_y[None, :]
    value[mask] = matrix[mask] * (np.log(matrix[mask] / denominators[mask]) / np.log(base))

    return float(value.sum())


def redundancy(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    """
    Redundancy R = 1 - H(X) / log_b(n).

    Measures deviation from uniformity.
    """
    p = _as_probability_vector(probabilities)
    n = p.size

    if n <= 1:
        return 0.0

    return float(1.0 - (entropy(p, base=base) / math.log(n, base)))


def kl_divergence(p: ArrayLike, q: ArrayLike, *, base: float = 2.0) -> float:
    """
    Kullback–Leibler divergence D_KL(P || Q).

    Requires:
        - P and Q valid probability vectors
        - Q strictly positive wherever P is positive
    """
    p_vec = _as_probability_vector(p)
    q_vec = _as_probability_vector(q, dimension=p_vec.size)

    mask = p_vec > 0.0

    if np.any((q_vec <= 0.0) & mask):
        raise ValueError("q must be strictly positive wherever p is strictly positive")

    return float(np.sum(p_vec[mask] * (np.log(p_vec[mask] / q_vec[mask]) / np.log(base))))


def js_divergence(p: ArrayLike, q: ArrayLike, *, base: float = 2.0) -> float:
    """
    Jensen–Shannon divergence:
        D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
        where M = (P + Q) / 2
    """
    p_vec = _as_probability_vector(p)
    q_vec = _as_probability_vector(q, dimension=p_vec.size)

    m = 0.5 * (p_vec + q_vec)

    return float(
        0.5 * kl_divergence(p_vec, m, base=base)
        + 0.5 * kl_divergence(q_vec, m, base=base)
    )


def spectral_entropy(signal: ArrayLike, *, base: float = 2.0) -> float:
    """
    Shannon entropy of normalized power spectrum of a 1D signal.

    Steps:
        - FFT magnitude
        - Normalize spectrum
        - Compute entropy
    """
    values = np.asarray(signal, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("signal must be a 1D array")

    if values.size == 0:
        raise ValueError("signal cannot be empty")

    spectrum = np.abs(np.fft.rfft(values))

    if np.all(spectrum == 0.0):
        return 0.0

    spectrum = spectrum / spectrum.sum()
    spectrum = spectrum[spectrum > 0.0]

    return float(-(spectrum * np.log(spectrum) / np.log(base)).sum())


def normalized_combination_entropy(probabilities: ArrayLike, *, omega_size: int, base: float = 2.0) -> float:
    """
    Normalized entropy H_norm(S) = H(S) / log_b(|Ω|).

    Ensures:
        - probabilities length matches omega_size
        - omega_size >= 2
    """
    if omega_size <= 1:
        raise ValueError("omega_size must be at least 2")

    p = _as_probability_vector(probabilities)

    if p.size != omega_size:
        raise ValueError(f"expected {omega_size} probabilities, received {p.size}")

    denom = math.log(omega_size, base)

    if denom == 0.0:
        raise ValueError("logarithm base results in a zero denominator for the provided omega size")

    return float(entropy(p, base=base) / denom)


def regime_transition_probability(p_a: ArrayLike, p_b: ArrayLike, *, tau: float = 0.05, base: float = 2.0) -> bool:
    """
    Detect regime transition when JS divergence exceeds threshold tau.
    """
    if tau < 0.0:
        raise ValueError("tau must be non-negative")

    return bool(js_divergence(p_a, p_b, base=base) > tau)


def monte_carlo_uniform_benchmark(
        *,
        domain_size: int = 10,
        samples_per_run: int = 1000,
        trials: int = 200,
        seed: int = 1337,
        base: float = 2.0,
) -> dict[str, float | int]:
    """
    Monte Carlo benchmark under uniform categorical H0.

    Computes:
        - empirical entropy distribution
        - JS divergence distribution
        - theoretical entropy log_b(domain_size)
    """
    if domain_size <= 1:
        raise ValueError("domain_size must be at least 2")

    if samples_per_run <= 0:
        raise ValueError("samples_per_run must be positive")

    rng = np.random.default_rng(seed)

    entropies: list[float] = []
    js_scores: list[float] = []

    for _ in range(trials):
        p = np.full(domain_size, 1.0 / domain_size, dtype=np.float64)
        sample = rng.multinomial(samples_per_run, p)
        empirical = sample / sample.sum()

        entropies.append(entropy(empirical, base=base))

        q = rng.dirichlet(np.ones(domain_size), size=1)[0]
        js_scores.append(js_divergence(empirical, q, base=base))

    theoretical = math.log(domain_size, base)

    return {
        "theoretical_entropy": theoretical,
        "mean_empirical_entropy": float(np.mean(entropies)),
        "std_empirical_entropy": float(np.std(entropies)),
        "mean_js_divergence": float(np.mean(js_scores)),
        "sample_size": samples_per_run,
        "trials": trials,
    }
