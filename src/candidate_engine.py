from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Sequence

import numpy as np


# A type alias for any 1D probability-like vector.
ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class Candidate:
    """
    Represents a candidate combination with its score, rank, and optional model-implied probability.
    """
    numbers: tuple[int, ...]
    score: float
    rank: int
    model_probability: float | None = None


@dataclass(frozen=True)
class BacktestWindow:
    """
    Represents a single walk-forward backtest window with metadata and evaluation metrics.
    """
    start_index: int
    end_index: int
    train_size: int
    test_size: int
    metric: float
    n_hits: int


def _validate_probability_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    """
    Validate and normalize a 1D probability vector.

    Ensures:
        - 1D shape
        - non-empty
        - finite values
        - non-negative
        - positive total mass

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


def generate_candidates(
        probabilities: ArrayLike,
        *,
        number_pool: Sequence[int],
        combination_size: int,
        limit: int = 10,
        exclude: Iterable[Sequence[int]] | None = None,
) -> list[Candidate]:
    """
    Generate high-mass candidate subsets using marginal inclusion probabilities.

    Strategy:
        - Sort numbers by descending probability.
        - Restrict candidate pool size to avoid combinatorial explosion.
        - Enumerate combinations and score them by summed probabilities.
        - Exclude user-specified combinations.
        - Return top-ranked candidates.
    """
    if combination_size <= 0:
        raise ValueError("combination_size must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    pool = tuple(sorted(int(value) for value in number_pool))
    if not pool:
        raise ValueError("number_pool cannot be empty")
    if combination_size > len(pool):
        raise ValueError("combination_size exceeds the number pool cardinality")

    probs = _validate_probability_vector(probabilities, name="probabilities")
    if probs.size != len(pool):
        raise ValueError("probabilities length must match the number pool size")

    excluded = {tuple(sorted(int(item) for item in combo)) for combo in (exclude or [])}

    ordering = np.argsort(-probs, kind="stable")

    candidate_pool_size = min(len(pool), max(combination_size, combination_size + limit - 1))

    while comb(candidate_pool_size, combination_size) > 250_000 and candidate_pool_size > combination_size:
        candidate_pool_size -= 1

    ranked_indices = ordering[:candidate_pool_size]

    possibilities = (
        (
            tuple(sorted(pool[int(ranked_indices[offset])] for offset in offsets)),
            float(sum(probs[int(ranked_indices[offset])] for offset in offsets)),
        )
        for offsets in combinations(range(candidate_pool_size), combination_size)
    )

    ranked_candidates = sorted(
        ((numbers, score) for numbers, score in possibilities if numbers not in excluded),
        key=lambda item: (-item[1], item[0]),
    )[:limit]

    selected = [
        Candidate(numbers=numbers, score=score, rank=rank)
        for rank, (numbers, score) in enumerate(ranked_candidates, start=1)
    ]

    if not selected:
        raise ValueError("failed to generate any valid candidates")

    return selected


def _log_elementary_symmetric_sum(log_weights: np.ndarray, subset_size: int) -> float:
    """
    Compute the log of the elementary symmetric sum of order k.

    Used for conditional Bernoulli subset probabilities.
    """
    if subset_size < 0 or subset_size > len(log_weights):
        return -np.inf

    values = np.full(subset_size + 1, -np.inf, dtype=np.float64)
    values[0] = 0.0

    processed = 0
    for log_weight in log_weights:
        processed += 1
        for size in range(min(processed, subset_size), 0, -1):
            values[size] = np.logaddexp(values[size], values[size - 1] + log_weight)

    return float(values[subset_size])


def fixed_size_subset_probability(
        inclusion_propensities: ArrayLike,
        selected_indices: Sequence[int],
        *,
        combination_size: int,
) -> float:
    """
    Compute the conditional Bernoulli probability of selecting exactly the given subset.

    Conditioning:
        - Independent Bernoulli propensities.
        - Fixed subset size constraint.
    """
    propensities = np.asarray(inclusion_propensities, dtype=np.float64)
    if propensities.ndim != 1 or propensities.size == 0:
        raise ValueError("inclusion_propensities must be a non-empty 1D vector")
    if np.any(~np.isfinite(propensities)) or np.any((propensities < 0.0) | (propensities > 1.0)):
        raise ValueError("inclusion_propensities must be finite values between 0 and 1")

    if combination_size <= 0 or combination_size > propensities.size:
        raise ValueError("combination_size must be within the propensity vector size")

    selected = tuple(int(index) for index in selected_indices)

    if (
            len(selected) != combination_size
            or len(selected) != len(set(selected))
            or any(index < 0 or index >= propensities.size for index in selected)
    ):
        raise ValueError("selected_indices must contain combination_size distinct valid indices")

    forced_in = set(np.flatnonzero(propensities == 1.0).tolist())
    if len(forced_in) > combination_size or not forced_in.issubset(selected):
        return 0.0

    remaining = combination_size - len(forced_in)

    available_indices = [
        index for index, value in enumerate(propensities)
        if value > 0.0 and index not in forced_in
    ]

    selected_remaining = [index for index in selected if index not in forced_in]

    if len(available_indices) < remaining or any(index not in available_indices for index in selected_remaining):
        return 0.0

    if remaining == 0:
        return 1.0

    log_weights = np.log(propensities[available_indices]) - np.log1p(-propensities[available_indices])
    index_to_log_weight = dict(zip(available_indices, log_weights, strict=True))

    log_normalizer = _log_elementary_symmetric_sum(log_weights, remaining)
    log_subset_weight = sum(index_to_log_weight[index] for index in selected_remaining)

    probability = float(np.exp(log_subset_weight - log_normalizer))
    return min(1.0, max(0.0, probability))


def generate_fixed_size_candidates(
        inclusion_propensities: ArrayLike,
        *,
        number_pool: Sequence[int],
        combination_size: int,
        limit: int = 10,
) -> list[Candidate]:
    """
    Generate and rank candidate combinations using conditional Bernoulli subset probabilities.

    Ranking:
        - Primary: model-implied subset probability.
        - Secondary: lexicographic order of numbers.
    """
    if combination_size <= 0:
        raise ValueError("combination_size must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    pool = tuple(int(value) for value in number_pool)
    if not pool or len(pool) != len(set(pool)):
        raise ValueError("number_pool must contain unique values")
    if combination_size > len(pool):
        raise ValueError("combination_size exceeds the number pool cardinality")

    propensities = np.asarray(inclusion_propensities, dtype=np.float64)
    if propensities.shape != (len(pool),):
        raise ValueError("inclusion_propensities length must match number_pool")
    if np.any(~np.isfinite(propensities)) or np.any((propensities < 0.0) | (propensities > 1.0)):
        raise ValueError("inclusion_propensities must be finite values between 0 and 1")

    forced_indices = np.flatnonzero(propensities == 1.0).tolist()
    if len(forced_indices) > combination_size:
        raise ValueError("more numbers have propensity 1 than the fixed combination size")

    remaining_size = combination_size - len(forced_indices)
    forced_set = set(forced_indices)

    eligible_indices = [
        index for index, value in enumerate(propensities)
        if value > 0.0 and index not in forced_set
    ]

    if len(eligible_indices) < remaining_size:
        raise ValueError("not enough positive-propensity numbers to fill a combination")

    if remaining_size == 0:
        numbers = tuple(sorted(pool[index] for index in forced_indices))
        score = float(sum(propensities[index] for index in forced_indices))
        return [Candidate(numbers=numbers, score=score, rank=1, model_probability=1.0)]

    log_weights = np.log(propensities[eligible_indices]) - np.log1p(-propensities[eligible_indices])
    log_weight_by_index = dict(zip(eligible_indices, log_weights, strict=True))

    log_normalizer = _log_elementary_symmetric_sum(log_weights, remaining_size)

    ordered_indices = sorted(
        eligible_indices,
        key=lambda index: (-log_weight_by_index[index], pool[index]),
    )

    candidate_pool_size = min(len(ordered_indices), remaining_size + limit - 1)

    while comb(candidate_pool_size, remaining_size) > 250_000 and candidate_pool_size > remaining_size:
        candidate_pool_size -= 1

    candidate_indices = ordered_indices[:candidate_pool_size]

    ranked: list[tuple[tuple[int, ...], float, float]] = []

    for selected_remaining in combinations(candidate_indices, remaining_size):
        selected = forced_indices + list(selected_remaining)
        numbers = tuple(sorted(pool[index] for index in selected))
        log_subset_weight = sum(log_weight_by_index[index] for index in selected_remaining)
        probability = float(np.exp(log_subset_weight - log_normalizer))
        score = float(sum(propensities[index] for index in selected))
        ranked.append((numbers, score, min(1.0, max(0.0, probability))))

    ranked.sort(key=lambda item: (-item[2], item[0]))

    return [
        Candidate(numbers=numbers, score=score, rank=rank, model_probability=probability)
        for rank, (numbers, score, probability) in enumerate(ranked[:limit], start=1)
    ]


def select_diverse_candidates(
        candidates: Sequence[Candidate],
        *,
        limit: int,
        overlap_penalty: float = 0.25,
        maximum_overlap: float = 1.0,
) -> list[Candidate]:
    """
    Greedy selection of diverse candidates.

    Penalizes:
        - high overlap with already selected candidates
        - redundant combinations

    Ensures:
        - unique combinations
        - finite scores
    """
    if not candidates:
        raise ValueError("candidates cannot be empty")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= overlap_penalty <= 1.0:
        raise ValueError("overlap_penalty must be between 0 and 1")
    if not 0.0 <= maximum_overlap <= 1.0:
        raise ValueError("maximum_overlap must be between 0 and 1")

    if any(not candidate.numbers for candidate in candidates):
        raise ValueError("candidate combinations cannot be empty")
    if any(not np.isfinite(candidate.score) for candidate in candidates):
        raise ValueError("candidate scores must be finite")

    unique: dict[tuple[int, ...], Candidate] = {}

    for candidate in candidates:
        numbers = tuple(sorted(set(int(value) for value in candidate.numbers)))
        if len(numbers) != len(candidate.numbers):
            raise ValueError("candidate combinations must not contain duplicate numbers")

        previous = unique.get(numbers)
        if previous is None or candidate.score > previous.score:
            unique[numbers] = Candidate(numbers, float(candidate.score), candidate.rank)

    remaining = list(unique.values())
    selected: list[Candidate] = []

    while remaining and len(selected) < limit:
        best: Candidate | None = None
        best_adjusted_score = -np.inf

        for candidate in remaining:
            overlaps = [
                len(set(candidate.numbers) & set(chosen.numbers))
                / max(1, len(set(candidate.numbers) | set(chosen.numbers)))
                for chosen in selected
            ]

            maximum_seen_overlap = max(overlaps, default=0.0)

            if maximum_seen_overlap > maximum_overlap:
                continue

            adjusted_score = candidate.score * (1.0 - overlap_penalty * maximum_seen_overlap)

            if adjusted_score > best_adjusted_score or (
                    adjusted_score == best_adjusted_score
                    and (best is None or candidate.numbers < best.numbers)
            ):
                best = candidate
                best_adjusted_score = adjusted_score

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)

    return [
        Candidate(numbers=item.numbers, score=item.score, rank=rank)
        for rank, item in enumerate(selected, start=1)
    ]


def candidate_score(
        actual_numbers: Sequence[int],
        candidate_numbers: Sequence[int],
        *,
        weights: Sequence[float] | None = None,
) -> float:
    """
    Score a candidate by weighted overlap with the actual draw.

    If weights are provided:
        - They must match candidate size.
        - They must be non-negative.

    Returns:
        float: Weighted overlap score in [0, 1].
    """
    actual = tuple(int(value) for value in actual_numbers)
    candidate = tuple(int(value) for value in candidate_numbers)

    if not actual or not candidate:
        raise ValueError("actual_numbers and candidate_numbers cannot be empty")

    if weights is None:
        weights = np.ones(len(candidate), dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (len(candidate),):
            raise ValueError("weights length must match candidate size")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")

    overlaps = np.array([1.0 if value in actual else 0.0 for value in candidate], dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)

    weighted_overlap = float(np.dot(overlaps, weights_array))
    return float(weighted_overlap / max(float(weights_array.sum()), 1e-12))


def walk_forward_backtest(
        series: Sequence[float],
        *,
        train_window: int,
        test_window: int,
        score_fn,
) -> list[BacktestWindow]:
    """
    Run a walk-forward backtest over a scalar series.

    For each window:
        - Train on train_window values.
        - Test on test_window values.
        - Compute metric using score_fn.
        - Count hits above training mean.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be a 1D array")
    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive")

    length = values.size
    if train_window + test_window > length:
        raise ValueError("train_window + test_window exceeds the series length")

    windows: list[BacktestWindow] = []

    for start in range(0, length - train_window - test_window + 1):
        train_values = values[start:start + train_window]
        test_values = values[start + train_window:start + train_window + test_window]

        metric = float(score_fn(train_values, test_values))

        windows.append(
            BacktestWindow(
                start_index=start,
                end_index=start + train_window + test_window - 1,
                train_size=train_window,
                test_size=test_window,
                metric=metric,
                n_hits=int(np.sum(test_values > np.mean(train_values))),
            )
        )

    return windows


def mean_absolute_error_score(train_values: Sequence[float], test_values: Sequence[float]) -> float:
    """
    Compute a deterministic MAE metric for backtesting.

    MAE = mean(|test[i] - train[i]|)

    Returns:
        float: MAE score.
    """
    train = np.asarray(train_values, dtype=np.float64)
    test = np.asarray(test_values, dtype=np.float64)

    if train.size == 0 or test.size == 0:
        raise ValueError("train and test values cannot be empty")

    if train.size > 0 and test.size > 0:
        return float(np.mean(np.abs(test - train[: test.size])))

    return 0.0
