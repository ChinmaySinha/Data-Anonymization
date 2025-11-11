import os
import json
from tqdm import tqdm  # For the progress bar
from detectors import detect_all_pii
from pii_evaluate import calculate_metrics

def evaluate_pipeline_performance(sample_text: str, ground_truth_labels: list):
    """
    NOTE: This function is no longer used by the main benchmark script,
    but it is kept here for your own single-sample testing if needed.
    """
    print("="*50)
    print("      RUNNING PIPELINE PERFORMANCE EVALUATION")
    print("="*50)
    # ... (rest of the function is not used in the main benchmark) ...


# --- THIS IS THE BENCHMARKING LOGIC ---
if __name__ == '__main__':
    print("Starting full benchmark evaluation...")
    
    # 3. Collect total TP, FP, and FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sample_count = 0

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Load the entire validation file
        # --- THIS LINE IS NOW CORRECTED (TYPO REMOVED) ---
        validation_file_path = os.path.join(
            script_dir, "pii-masking-300k", "data", "validation", "1english_openpii_8k.jsonl"
        )
        
        if not os.path.exists(validation_file_path):
            print(f"FATAL: Validation file not found at {validation_file_path}")
            print("Please check the path and folder names.")
        else:
            print(f"Loading validation file: {validation_file_path}")
            
            # --- 2. LOOP WITH TQDM PROGRESS BAR ---
            # First, get the total number of lines for the progress bar
            with open(validation_file_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for line in f)

            # Re-open the file and process it with tqdm
            with open(validation_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=num_lines, desc="Evaluating Benchmark"):
                    try:
                        sample = json.loads(line)
                        original_text = sample['source_text']
                        ground_truth = sample['privacy_mask']

                        # Format ground truth from the .jsonl format
                        formatted_ground_truth = [
                            {'start': item['start'], 'end': item['end'], 'entity_group': item['label']}
                            for item in ground_truth
                        ]

                        # Run the full detection pipeline (now quiet)
                        detected_entities = detect_all_pii(original_text)
                        
                        # Get metrics for this single sample
                        metrics = calculate_metrics(
                            predictions=detected_entities, 
                            ground_truth=formatted_ground_truth
                        )
                        
                        # Aggregate the totals
                        total_tp += metrics['true_positives']
                        total_fp += metrics['false_positives']
                        total_fn += metrics['false_negatives']
                        
                        sample_count += 1

                    except (json.JSONDecodeError, KeyError):
                        # Silently skip bad lines
                        continue

            print(f"\n\nBenchmark complete. Processed a total of {sample_count} samples.")

            # 4. Calculate a final F1-score from those totals
            print("\n" + "="*50)
            print("      FINAL BENCHMARK REPORT")
            print("="*50)
            print(f"\n- Total True Positives: {total_tp}")
            print(f"- Total False Positives: {total_fp}")
            print(f"- Total False Negatives: {total_fn}")
            print("-" * 50)

            # Calculate final metrics from the aggregated totals
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"- Final Precision: {precision:.4f}")
            print(f"- Final Recall: {recall:.4f}")
            print(f"- Final F1-Score: {f1_score:.4f}")
            print("="*50)

    except Exception as e:
        print(f"\nAn error occurred during benchmarking: {e}")