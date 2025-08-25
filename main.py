import os
from datasets import load_dataset
from detectors import detect_all_pii # Import the new all-in-one detector
from sensitivity_classifier import classify_entity_sensitivity
from anonymization_module import anonymize_text

if __name__ == '__main__':
    user_input = input("Enter the text you want to anonymize (or press Enter to use a default sample):\n")

    if user_input.strip():
        original_text = user_input
    else:
        print("\nNo input provided. Using default sample from the dataset...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_dataset_path = os.path.join(script_dir, "pii-masking-300k")
        dataset_dict = load_dataset(local_dataset_path)
        sample = dataset_dict['train'][0]
        original_text = sample['source_text']
        
    # --- The Simplified Pipeline ---
    
    # Step 1: Detect all potential PII entities using our expert committee
    detected_entities = detect_all_pii(original_text)
    
    if not detected_entities:
        print("No PII entities detected.")
    else:
        # Step 2: Classify the sensitivity of each detected entity
        print("\n--- 2. Classifying entity sensitivity... ---")
        entities_with_sensitivity = []
        for entity in detected_entities:
            entity_text = entity['word']
            sensitivity = classify_entity_sensitivity(entity_text)
            entity['sensitivity'] = sensitivity
            entities_with_sensitivity.append(entity)
            print(f"  - Entity: '{entity_text}' ({entity['entity_group']}) -> Sensitivity: {sensitivity}")

        # Step 3: Anonymize the text based on the classified sensitivities
        print("\n--- 3. Anonymizing text based on sensitivity... ---")
        final_anonymized_text = anonymize_text(original_text, entities_with_sensitivity)
        
        # Step 4: Display the final results
        print("\n" + "="*40)
        print("      FINAL ANONYMIZATION RESULT")
        print("="*40)
        print("\n--- Original Text ---")
        print(original_text)
        print("\n--- Anonymized Text ---")
        print(final_anonymized_text)
        print("\n" + "="*40)
