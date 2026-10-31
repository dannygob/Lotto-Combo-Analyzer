import math

import numpy as np

from src.information_theory import (
    conditional_entropy,
    entropy,
    js_divergence,
    monte_carlo_uniform_benchmark,
    mutual_information,
    normalized_combination_entropy,
    redundancy,
    regime_transition_probability,
    spectral_entropy,
)


def test_uniform_distribution_has_max_entropy() -> None:
    p = np.full(8, 1.0 / 8)
    assert math.isclose(entropy(p), 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(redundancy(p), 0.0, rel_tol=1e-9, abs_tol=1e-9)


def test_one_hot_distribution_has_zero_entropy() -> None:
    p = np.array([1.0, 0.0, 0.0])
    assert entropy(p) == 0.0


def test_mutual_information_zero_for_independent_joint_distribution() -> None:
    p_x = np.array([0.5, 0.5])
    p_y = np.array([0.5, 0.5])
    joint = np.outer(p_x, p_y)
    assert math.isclose(mutual_information(joint), 0.0, abs_tol=1e-12)


def test_js_divergence_is_zero_for_identical_distributions() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert js_divergence(p, p) == 0.0


def test_conditional_entropy_matches_definition() -> None:
    joint = np.array([[0.25, 0.25], [0.25, 0.25]])
    assert math.isclose(conditional_entropy(joint), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_spectral_entropy_returns_finite_value() -> None:
    signal = np.sin(np.linspace(0.0, 20.0, 256))
    value = spectral_entropy(signal)
    assert value >= 0.0
    assert math.isfinite(value)


def test_normalized_combination_entropy_uses_omega_size() -> None:
    p = np.array([0.5, 0.25, 0.25])
    value = normalized_combination_entropy(p, omega_size=3)
    assert 0.0 <= value <= 1.0


def test_regime_transition_threshold_works() -> None:
    p_a = np.array([0.8, 0.2])
    p_b = np.array([0.2, 0.8])
    assert regime_transition_probability(p_a, p_b, tau=0.05)


def test_monte_carlo_benchmark_matches_uniform_null() -> None:
    summary = monte_carlo_uniform_benchmark(domain_size=4, samples_per_run=1000, trials=20, seed=2)
    assert summary["theoretical_entropy"] == 2.0
    assert summary["mean_empirical_entropy"] > 0.0
    assert summary["trials"] == 20
