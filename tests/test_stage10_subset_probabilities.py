from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.candidate_engine import (
    fixed_size_subset_probability,
    generate_fixed_size_candidates,
)


def test_uniform_propensities_assign_equal_mass_to_every_fixed_size_subset() -> None:
    propensities = np.full(5, 0.4)
    subsets = list(itertools.combinations(range(5), 2))
    probabilities = [
        fixed_size_subset_probability(
            propensities,
            subset,
            combination_size=2,
        )
        for subset in subsets
    ]

    assert len(probabilities) == 10
    assert all(math.isclose(value, 0.1, abs_tol=1e-12) for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)


def test_fixed_size_subset_distribution_normalizes_and_ranks_candidates() -> None:
    propensities = np.array([0.8, 0.6, 0.3, 0.1])
    all_subsets = list(itertools.combinations(range(4), 2))
    probabilities = [
        fixed_size_subset_probability(
            propensities,
            subset,
            combination_size=2,
        )
        for subset in all_subsets
    ]
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)

    candidates = generate_fixed_size_candidates(
        propensities,
        number_pool=[1, 2, 3, 4],
        combination_size=2,
        limit=6,
    )
    assert len(candidates) == 6
    assert candidates[0].numbers == (1, 2)
    assert all(
        candidates[index].model_probability >= candidates[index + 1].model_probability
        for index in range(len(candidates) - 1)
    )
    assert math.isclose(
        sum(candidate.model_probability for candidate in candidates),
        1.0,
        abs_tol=1e-12,
    )


def test_fixed_size_subset_probability_respects_forced_and_impossible_numbers() -> None:
    propensities = np.array([1.0, 0.6, 0.4, 0.0])
    assert math.isclose(
        fixed_size_subset_probability(propensities, [0, 1], combination_size=2),
        9.0 / 13.0,
        abs_tol=1e-12,
    )
    assert fixed_size_subset_probability(
        propensities,
        [1, 2],
        combination_size=2,
    ) == 0.0

    candidates = generate_fixed_size_candidates(
        propensities,
        number_pool=[1, 2, 3, 4],
        combination_size=2,
        limit=4,
    )
    assert len(candidates) == 2
    assert candidates[0].numbers == (1, 2)
    assert candidates[1].numbers == (1, 3)
    assert math.isclose(
        sum(candidate.model_probability for candidate in candidates),
        1.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    ("propensities", "selected", "combination_size"),
    [
        ([0.2, float("nan")], [0], 1),
        ([0.2, -0.1], [0], 1),
        ([0.2, 1.1], [0], 1),
        ([0.2, 0.3], [0, 0], 2),
        ([0.2, 0.3], [2], 1),
        ([0.2, 0.3], [0], 3),
    ],
)
def test_fixed_size_subset_probability_rejects_invalid_inputs(
    propensities: list[float],
    selected: list[int],
    combination_size: int,
) -> None:
    with pytest.raises(ValueError):
        fixed_size_subset_probability(
            propensities,
            selected,
            combination_size=combination_size,
        )
