# ============================================================================
# MODULE: Sum Distribution and Model Improvements
# ============================================================================
import numpy as np
from typing import Dict, Any


def calculate_sum_distribution(pool_id: str, minimum: int, maximum: int, numbers_drawn: int) -> dict:
    """
    Calculate the expected sum distribution for a pool.
    Includes minimum/maximum possible sum, expected mean, standard deviation,
    and helper ranges for ±1σ and ±2σ.
    """
    # Minimum and maximum possible sum for a combination of size numbers_drawn
    min_sum = sum(range(1, numbers_drawn + 1))
    max_sum = sum(range(maximum - numbers_drawn + 1, maximum + 1))

    # Expected sum assuming uniform distribution
    expected_sum = (minimum + maximum) * numbers_drawn / 2

    # Variance formula for sum of uniform discrete variables
    # NOTE: This formula assumes sampling without replacement.
    variance = (
            numbers_drawn
            * (maximum - minimum + 1) ** 2
            * (maximum - minimum + 1 - numbers_drawn)
            / ((maximum - minimum + 1) ** 2 * (maximum - minimum))
    )
    std_dev = variance ** 0.5

    return {
        "pool_id": pool_id,
        "minimum_sum": min_sum,
        "maximum_sum": max_sum,
        "expected_sum": expected_sum,
        "std_dev": std_dev,
        "one_std_range": [expected_sum - std_dev, expected_sum + std_dev],
        "two_std_range": [expected_sum - 2 * std_dev, expected_sum + 2 * std_dev],
        # Helper function to check if a sum falls within ±2σ
        "is_within_normal_range": lambda s: (expected_sum - 2 * std_dev) <= s <= (expected_sum + 2 * std_dev),
    }


def filter_combinations_by_sum(combinations: list, pool_id: str, sum_distribution: dict) -> dict:
    """
    Filter combinations based on their sum relative to the pool's sum distribution.
    Rejects combinations outside the min/max range or outside ±2σ.
    """
    filtered = {
        "accepted": [],
        "rejected": [],
        "rejection_reasons": {},
    }

    min_s = sum_distribution["minimum_sum"]
    max_s = sum_distribution["maximum_sum"]
    expected = sum_distribution["expected_sum"]
    std = sum_distribution["std_dev"]

    for combo in combinations:
        combo_sum = sum(combo)
        combo_id = f"{pool_id}_{'_'.join(map(str, combo))}"

        # Reject sums outside absolute min/max
        if combo_sum < min_s or combo_sum > max_s:
            filtered["rejected"].append(combo)
            filtered["rejection_reasons"][combo_id] = (
                f"Sum {combo_sum} outside allowed range [{min_s}-{max_s}]"
            )
            continue

        # Reject sums outside ±2σ
        if not sum_distribution["is_within_normal_range"](combo_sum):
            filtered["rejected"].append(combo)
            filtered["rejection_reasons"][combo_id] = (
                f"Sum {combo_sum} outside 2σ ({expected:.1f} ± {2 * std:.1f})"
            )
            continue

        # Accept combination
        filtered["accepted"].append({
            "combination": combo,
            "sum": combo_sum,
            "deviation_from_expected": abs(combo_sum - expected),
            "z_score": abs(combo_sum - expected) / std if std > 0 else 0,
        })

    # Sort accepted combinations by closeness to expected sum
    filtered["accepted"].sort(key=lambda x: x["deviation_from_expected"])

    return filtered


def apply_model_improvements(metrics: dict, pool_metrics: dict, predictions: dict, ruleset_pools: list) -> dict:
    """
    Apply all model improvements:
    - Pool weighting based on performance
    - Threshold adjustment based on calibration/drift
    - Sum distribution validation
    - Weighted prediction ranking
    """
    from src.csv_pool_metrics import (
        adjust_pool_weights_by_performance,
        adjust_prediction_threshold_by_calibration,
    )

    # Compute pool weights from performance metrics
    pool_weights = adjust_pool_weights_by_performance(pool_metrics)

    # Calibration-based threshold adjustment
    calibration_passed = metrics.get("calibration_passed", False)
    drift_within_tolerance = metrics.get("drift_within_tolerance", True)
    beats_null = metrics.get("beats_null", False)

    threshold = adjust_prediction_threshold_by_calibration(
        calibration_passed,
        drift_within_tolerance,
        beats_null
    )

    # Compute sum distributions for each pool
    sum_distributions = {}
    for pool in ruleset_pools:
        sum_distributions[pool.pool_id] = calculate_sum_distribution(
            pool.pool_id, pool.minimum, pool.maximum, pool.numbers_drawn
        )

    adjusted_predictions = {}

    # Apply pool weights and threshold to predictions
    for pool_id, pool_preds in predictions.items():
        weight = pool_weights.get(pool_id, 1.0 / len(pool_metrics)) if pool_metrics else 1.0

        adjusted_pool_preds = []
        for pred in pool_preds:
            original_prob = pred.get("probability", 0.0)
            weighted_prob = original_prob * weight
            binary_pred = 1 if weighted_prob >= threshold else 0
            confidence = abs(weighted_prob - threshold)

            adjusted_pool_preds.append({
                "number": pred.get("number"),
                "original_probability": original_prob,
                "weighted_probability": weighted_prob,
                "binary_prediction": binary_pred,
                "confidence": confidence,
                "pool_weight": weight,
                "threshold_used": threshold,
            })

        # Sort by weighted probability
        adjusted_pool_preds.sort(key=lambda x: x["weighted_probability"], reverse=True)

        # Trim to pool size
        pool_info = next((p for p in ruleset_pools if p.pool_id == pool_id), None)
        if pool_info:
            adjusted_pool_preds = adjusted_pool_preds[:pool_info.numbers_drawn]

        adjusted_predictions[pool_id] = adjusted_pool_preds

    # Validate sum of selected numbers
    sum_validations = {}
    for pool_id, pool_preds in adjusted_predictions.items():
        combo = [p["number"] for p in pool_preds if p["binary_prediction"] == 1]

        if len(combo) >= 2:
            combo_sum = sum(combo)
            sum_dist = sum_distributions.get(pool_id, {})

            expected = sum_dist.get("expected_sum", 0)
            std = sum_dist.get("std_dev", 1)

            sum_validations[pool_id] = {
                "combination": combo,
                "sum": combo_sum,
                "expected_sum": expected,
                "std_dev": std,
                "z_score": abs(combo_sum - expected) / std if std > 0 else 0,
                "is_normal": sum_dist.get("is_within_normal_range", lambda s: True)(combo_sum),
            }

    return {
        "adjusted_predictions": adjusted_predictions,
        "pool_weights": pool_weights,
        "prediction_threshold": threshold,
        "sum_distributions": sum_distributions,
        "sum_validations": sum_validations,
        "calibration_applied": not calibration_passed,
        "overfitting_detected": not drift_within_tolerance,
    }
