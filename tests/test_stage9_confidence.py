from __future__ import annotations

import numpy as np
import pytest

from src.candidate_engine import Candidate, generate_candidates, select_diverse_candidates
from src.confidence_engine import probabilistic_diagnostics
from src.feature_engineering import compare_feature_distributions
from src.model_suite import adaptive_ensemble_weights


def test_probabilistic_diagnostics_benchmarks_uniform_null_and_regimes() -> None:
    probabilities = np.tile([0.25, 0.25, 0.25, 0.25], (40, 1))
    outcomes = np.arange(40, dtype=int) % 4
    report = probabilistic_diagnostics(
        probabilities,
        outcomes,
        regimes=["early"] * 20 + ["late"] * 20,
        bootstrap_samples=200,
        seed=17,
    )
    assert report["brier_score"] == pytest.approx(0.75)
    assert report["log_loss"] == pytest.approx(np.log(4.0))
    assert report["expected_calibration_error"] == pytest.approx(0.0)
    assert report["beats_uniform_at_95pct"] is False
    assert set(report["by_regime"]) == {"early", "late"}


def test_probabilistic_diagnostics_detects_clear_out_of_sample_lift() -> None:
    probabilities = np.tile([0.8, 0.1, 0.1], (30, 1))
    outcomes = np.zeros(30, dtype=int)
    report = probabilistic_diagnostics(probabilities, outcomes, bootstrap_samples=200)
    assert report["beats_uniform_at_95pct"] is True
    assert report["log_loss"] < report["uniform_log_loss"]


def test_adaptive_weights_favor_recently_better_model() -> None:
    weights = adaptive_ensemble_weights(
        [[0.4, 0.4], [0.8, 0.1]],
        temperature=0.2,
        recency_decay=0.5,
    )
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)
    assert weights[1] > weights[0]


def test_feature_drift_uses_distribution_distance() -> None:
    stable = compare_feature_distributions([0, 0, 1, 1], [0, 0, 1, 1], bins=4)
    shifted = compare_feature_distributions([0, 0, 1, 1], [10, 10, 11, 11], bins=4)
    assert stable.divergence == pytest.approx(0.0)
    assert stable.drift_detected is False
    assert shifted.drift_detected is True
    assert shifted.divergence > stable.divergence


def test_candidate_generation_is_unique_and_portfolio_penalizes_overlap() -> None:
    candidates = generate_candidates(
        [0.4, 0.3, 0.2, 0.1],
        number_pool=[1, 2, 3, 4],
        combination_size=2,
        limit=4,
    )
    assert len({candidate.numbers for candidate in candidates}) == 4
    portfolio = select_diverse_candidates(
        [
            Candidate((1, 2), 0.9, 1),
            Candidate((1, 3), 0.8, 2),
            Candidate((5, 6), 0.7, 3),
        ],
        limit=2,
        overlap_penalty=0.5,
    )
    assert portfolio[0].numbers == (1, 2)
    assert portfolio[1].numbers == (5, 6)
