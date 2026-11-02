# ============================================================================
# MODULE: Pool Metrics
# ============================================================================
import numpy as np
from collections import Counter
from typing import Dict, Any


def calculate_pool_metrics(all_probabilities_by_pool, all_outcomes_by_pool, ruleset):
    """
    Compute evaluation metrics for each pool independently.
    Metrics include MAE, MSE, RMSE, precision, recall, F1, log-loss, Brier score,
    and confusion matrix components.
    """
    pool_metrics = {}

    for pool in ruleset.pools:
        pool_id = pool.pool_id
        probabilities = all_probabilities_by_pool.get(pool_id, [])
        outcomes = all_outcomes_by_pool.get(pool_id, [])

        # Handle empty pools gracefully
        if len(probabilities) == 0 or len(outcomes) == 0:
            pool_metrics[pool_id] = {
                "mae": 0.0, "mse": 0.0, "rmse": 0.0,
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "log_loss": 0.0, "brier_score": 0.0,
                "true_positives": 0, "false_positives": 0,
                "false_negatives": 0, "true_negatives": 0,
                "accuracy": 0.0,
            }
            continue

        # Numerical stability for log-loss
        epsilon = 1e-15
        probabilities_clipped = np.clip(probabilities, epsilon, 1 - epsilon)

        # Convert outcomes to binary classification
        y_true_binary = (outcomes > 0).astype(int)
        y_pred_binary = (probabilities > 0.5).astype(int)

        # Confusion matrix components
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

        # Classification metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Regression-style metrics
        mae = float(np.mean(np.abs(probabilities - outcomes)))
        mse = float(np.mean((probabilities - outcomes) ** 2))
        rmse = float(np.sqrt(mse))

        # Probabilistic metrics
        log_loss = float(-np.mean(outcomes * np.log(probabilities_clipped) +
                                  (1 - outcomes) * np.log(1 - probabilities_clipped)))
        brier_score = float(np.mean((probabilities - outcomes) ** 2))

        pool_metrics[pool_id] = {
            "mae": mae, "mse": mse, "rmse": rmse,
            "precision": float(precision), "recall": float(recall), "f1_score": float(f1_score),
            "log_loss": log_loss, "brier_score": brier_score,
            "true_positives": int(tp), "false_positives": int(fp),
            "false_negatives": int(fn), "true_negatives": int(tn),
            "accuracy": float(accuracy),
        }

    return pool_metrics


def calculate_pool_frequency_data(frame, pool_columns, ruleset):
    """
    Compute frequency statistics for each pool.
    Counts how often each number appears and normalizes frequencies.
    """
    pool_frequency = {}

    for pool in ruleset.pools:
        pool_id = pool.pool_id
        columns = pool_columns.get(pool_id, [])

        if not columns:
            continue

        # Collect all numbers drawn in this pool
        all_numbers = []
        for col in columns:
            col_data = frame[col].dropna().tolist()
            all_numbers.extend(col_data)

        freq_counter = Counter(all_numbers)
        total_count = sum(freq_counter.values())

        # Build normalized frequency table
        frequencies = {}
        for num in range(pool.minimum, pool.maximum + 1):
            count = freq_counter.get(num, 0)
            frequencies[str(num)] = {
                "count": count,
                "frequency": count / total_count if total_count > 0 else 0.0,
            }

        pool_frequency[pool_id] = {
            "pool_name": pool_id,
            "numbers_drawn": pool.numbers_drawn,
            "minimum": pool.minimum,
            "maximum": pool.maximum,
            "total_numbers": len(all_numbers),
            "unique_numbers": len(freq_counter),
            "frequencies": frequencies,
        }

    return pool_frequency


def adjust_pool_weights_by_performance(pool_metrics: dict) -> dict:
    """
    Compute pool weights based on performance metrics.
    Weighted combination of F1, precision, and accuracy.
    """
    pool_weights = {}
    total_score = 0.0

    for pool_id, metrics in pool_metrics.items():
        f1 = metrics.get("f1_score", 0.0)
        precision = metrics.get("precision", 0.0)
        accuracy = metrics.get("accuracy", 0.0)

        # Weighted scoring formula
        score = (f1 * 0.5) + (precision * 0.3) + (accuracy * 0.2)

        # Minimum weight to avoid zeroing out pools
        score = max(score, 0.1)

        pool_weights[pool_id] = score
        total_score += score

    # Normalize weights
    if total_score > 0:
        for pool_id in pool_weights:
            pool_weights[pool_id] /= total_score
    else:
        # Fallback: uniform weights
        num_pools = len(pool_metrics)
        for pool_id in pool_weights:
            pool_weights[pool_id] = 1.0 / num_pools

    return pool_weights


def adjust_prediction_threshold_by_calibration(calibration_passed: bool,
                                               drift_within_tolerance: bool,
                                               beats_null: bool) -> float:
    """
    Adjust the prediction threshold based on model calibration quality.
    Threshold increases when calibration fails, drift is detected,
    or the model does not outperform the null baseline.
    """
    base_threshold = 0.5

    if not calibration_passed:
        base_threshold += 0.15
    if not drift_within_tolerance:
        base_threshold += 0.10
    if not beats_null:
        base_threshold += 0.20

    # Clamp threshold to safe range
    return max(0.3, min(0.95, base_threshold))
