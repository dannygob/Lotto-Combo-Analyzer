# evaluation.py

import pandas as pd

def evaluate_group_accuracy(predicted_combos, actual_combos):
    """
    Evaluate prediction accuracy at the group level (6-number combinations).

    Parameters:
        predicted_combos (list of tuples): Predicted combinations.
        actual_combos (list of tuples): Actual historical combinations.

    Returns:
        dict: Accuracy metrics including exact matches and partial overlaps.
    """
    exact_matches = 0
    partial_matches = {6: 0, 5: 0, 4: 0, 3: 0}

    for pred in predicted_combos:
        for actual in actual_combos:
            overlap = len(set(pred) & set(actual))
            if overlap == 6:
                exact_matches += 1
            elif overlap in partial_matches:
                partial_matches[overlap] += 1

    total = len(predicted_combos)
    metrics = {
        "total_predictions": total,
        "exact_match_count": exact_matches,
        "exact_match_percent": round((exact_matches / total) * 100, 2),
        "partial_match_counts": partial_matches,
        "partial_match_percent": {k: round((v / total) * 100, 2) for k, v in partial_matches.items()}
    }
    return metrics

def print_group_accuracy_report(metrics):
    """
    Print a formatted report of group-level prediction accuracy.

    Parameters:
        metrics (dict): Accuracy metrics from evaluate_group_accuracy().
    """
    print("🎯 Group-Level Prediction Accuracy Report")
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Exact Matches (6/6): {metrics['exact_match_count']} ({metrics['exact_match_percent']}%)")
    for k in sorted(metrics['partial_match_counts'].keys(), reverse=True):
        count = metrics['partial_match_counts'][k]
        percent = metrics['partial_match_percent'][k]
        print(f"Partial Matches ({k}/6): {count} ({percent}%)")
