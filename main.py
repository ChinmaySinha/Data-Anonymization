import json
import os
from datasets import load_dataset
from pii_evaluate import calculate_metrics
from regex_matcher import detect_pii_with_regex
from ner_detector import detect_entities_with_ner

def run_anonymization_pipeline(text: str) -> list:
    """Runs the full detection pipeline on a single text."""
    regex_results = detect_pii_with_regex(text)
    ner_results = detect_entities_with_ner(text)
    
    all_detections = regex_results + ner_results
    #all_detections = ner_results #checking just the ner capabilities
    # Remove duplicates that might arise from both methods
    unique_detections = []
    seen_positions = set()
    for item in sorted(all_detections, key=lambda x: x['start']):
        pos = (item['start'], item['end'])
        if pos not in seen_positions:
            unique_detections.append(item)
            seen_positions.add(pos)
    return unique_detections

if __name__ == '__main__':
    # --- MODIFIED CODE TO LOAD LOCAL DATASET ---
    # Get the path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the local dataset folder
    local_dataset_path = os.path.join(script_dir, "pii-masking-300k")
    
    print(f"Loading dataset from local path: {local_dataset_path}")
    # Load the dataset directly from the folder
    dataset_dict = load_dataset(local_dataset_path)
    # -------------------------------------------

    train_dataset = dataset_dict['train']
    
    # --- Process a Subset of the Dataset and Evaluate ---
    num_samples_to_process = 200
    subset = train_dataset.select(range(num_samples_to_process))

    total_tp, total_fp, total_fn = 0, 0, 0
    all_results = []

    print(f"\n--- 🚀 Starting Evaluation on {num_samples_to_process} Samples ---")
    for i, sample in enumerate(subset):
        print(f"\nProcessing sample {i+1}/{num_samples_to_process}...")
        
        # The keys 'source_text' and 'privacy_mask' are correct based on our previous debugging
        input_text = sample['source_text']
        ground_truth = sample['privacy_mask']

        predictions = run_anonymization_pipeline(input_text)
        metrics = calculate_metrics(predictions, ground_truth)
        
        total_tp += metrics['true_positives']
        total_fp += metrics['false_positives']
        total_fn += metrics['false_negatives']
        
        all_results.append({
            "id": sample.get("id", i),
            "text": input_text,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "metrics": metrics
        })

    # --- Calculate Final Metrics ---
    final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    final_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    final_f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

    print("\n--- ✅ Evaluation Finished ---")
    print("\n--- Overall Performance Metrics ---")
    print(f"  - Precision: {final_precision:.4f}")
    print(f"  - Recall:    {final_recall:.4f}")
    print(f"  - F1-Score:  {final_f1_score:.4f}")
    print("---------------------------------")
    
    # Save the detailed results to a file
    output_file_path = "evaluation_results.json"
    with open(output_file_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n🎉 Detailed results for each sample saved to '{output_file_path}'")