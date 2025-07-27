def calculate_metrics(predictions: list, ground_truth: list):
    """
    Calculates precision, recall, and F1-score for PII detection.

    Args:
        predictions: A list of dicts from our pipeline, e.g., [{'start': 10, 'end': 15, 'entity_group': 'PER'}].
        ground_truth: A list of dicts from the dataset's 'privacy_mask' column.

    Returns:
        A dictionary containing TP, FP, FN, precision, recall, and f1_score.
    """
    # Create sets of tuples for easy comparison. We only care about the span (start, end).
    pred_spans = set((p['start'], p['end']) for p in predictions)
    true_spans = set((gt['start'], gt['end']) for gt in ground_truth)

    true_positives = len(pred_spans.intersection(true_spans))
    false_positives = len(pred_spans.difference(true_spans))
    
    # --- THIS LINE IS NOW CORRECTED ---
    false_negatives = len(true_spans.difference(pred_spans))

    # Calculate metrics, handling division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }