import math

import numpy as np

from src.candidate_engine import (
    candidate_score,
    generate_candidates,
    mean_absolute_error_score,
    walk_forward_backtest,
)


def test_candidate_generation_uses_top_probability_mass() -> None:
    candidates = generate_candidates(
        np.array([0.35, 0.3, 0.2, 0.15]),
        number_pool=[1, 2, 3, 4],
        combination_size=2,
        limit=3,
    )
    assert len(candidates) == 3
    assert candidates[0].numbers == (1, 2)
    assert candidates[0].score >= 0.0


def test_candidate_score_uses_overlap_weighting() -> None:
    score = candidate_score([2, 4, 7], [1, 2, 4], weights=[0.2, 0.8, 0.0])
    assert math.isclose(score, 0.8, rel_tol=1e-12, abs_tol=1e-12)


def test_walk_forward_backtest_produces_windows() -> None:
    windows = walk_forward_backtest(np.linspace(0.0, 10.0, 20), train_window=6, test_window=2, score_fn=mean_absolute_error_score)
    assert len(windows) >= 1
    assert windows[0].train_size == 6
    assert windows[0].test_size == 2
    assert windows[0].metric >= 0.0


def test_candidate_generator_rejects_invalid_limits() -> None:
    try:
        generate_candidates(np.array([0.5, 0.5]), number_pool=[1, 2], combination_size=2, limit=0)
        raise AssertionError('limit=0 should have raised an error')
    except ValueError:
        pass


def test_candidate_score_handles_equal_weight_case() -> None:
    score = candidate_score([3, 5], [1, 3])
    assert math.isclose(score, 0.5, rel_tol=1e-12, abs_tol=1e-12)
