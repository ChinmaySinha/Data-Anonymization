import os
from datasets import load_dataset
from detectors import detect_all_pii
from pii_evaluate import calculate_metrics

def evaluate_pipeline_performance(sample_text: str, ground_truth_labels: list):
    """
    Runs the PII detection pipeline on a sample text and evaluates its performance
    against ground truth labels.

    Args:
        sample_text: The text to be analyzed.
        ground_truth_labels: A list of dictionaries, where each dictionary
                             represents a true PII entity's location and type.
    """
    print("="*50)
    print("      RUNNING PIPELINE PERFORMANCE EVALUATION")
    print("="*50)

    # Step 1: Run our hybrid detection pipeline on the sample text
    print("\n--- Detecting PII with the hybrid engine... ---")
    detected_entities = detect_all_pii(sample_text)
    print(f"\nDetected {len(detected_entities)} entities.")

    # Step 2: Calculate performance metrics
    print("\n--- Calculating performance metrics... ---")
    metrics = calculate_metrics(predictions=detected_entities, ground_truth=ground_truth_labels)

    # Step 3: Print the performance report
    print("\n" + "="*50)
    print("      PERFORMANCE EVALUATION REPORT")
    print("="*50)
    print(f"\n- True Positives (Correctly Found): {metrics['true_positives']}")
    print(f"- False Positives (Incorrectly Found): {metrics['false_positives']}")
    print(f"- False Negatives (Missed): {metrics['false_negatives']}")
    print("-" * 50)
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall: {metrics['recall']:.4f}")
    print(f"- F1-Score: {metrics['f1_score']:.4f}")
    print("="*50)

if __name__ == '__main__':
    # Load a default sample from the dataset to evaluate
    print("Loading a default sample from the dataset for evaluation...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_dataset_path = os.path.join(script_dir, "pii-masking-300k")
        dataset_dict = load_dataset(local_dataset_path, split='train', streaming=True)
        sample = next(iter(dataset_dict))

        original_text = sample['source_text']
        ground_truth = sample['privacy_mask']

        # The privacy_mask from the dataset is a list of lists, so we need to flatten it
        # and ensure it has the right keys for our calculate_metrics function.
        formatted_ground_truth = [
            {'start': item[1], 'end': item[2], 'entity_group': item[0]}
            for item in ground_truth
        ]

        evaluate_pipeline_performance(
            sample_text=original_text,
            ground_truth_labels=formatted_ground_truth
        )

    except Exception as e:
        print(f"\nCould not load the dataset. Please ensure the 'pii-masking-300k' dataset is available. Error: {e}")
