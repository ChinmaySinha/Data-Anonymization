import unittest
from detectors import detect_all_pii
from sensitivity_classifier import classify_entity_sensitivity
from anonymization_module import anonymize_text

class TestAnonymizationPipeline(unittest.TestCase):

    def test_full_pipeline_with_faker(self):
        """
        Tests the full PII anonymization pipeline with Faker producing dynamic fake data.
        """
        # Sample text containing various PII entities
        original_text = "Contact Mr. John Doe at john.doe@example.com. His Polish ID is ABC123456 and his location is New York."

        # A list of the original PII values we expect to be replaced
        original_pii_values = ["John Doe", "john.doe@example.com", "ABC123456", "New York", "Polish"]

        # --- Step 1: PII Detection ---
        detected_entities = detect_all_pii(original_text)

        # --- Step 2: Sensitivity Classification ---
        entities_with_sensitivity = []
        for entity in detected_entities:
            sensitivity = classify_entity_sensitivity(entity['entity_group'])
            entity['sensitivity'] = sensitivity
            entities_with_sensitivity.append(entity)

        # --- Step 3: Anonymization ---
        anonymized_text_result = anonymize_text(original_text, entities_with_sensitivity)

        # --- Step 4: Verification for Dynamic Data ---
        # Since Faker generates random data, we can't do an exact string comparison.
        # Instead, we verify that the original PII values are no longer in the output.

        print("\n--- Test Pipeline Results (with Faker) ---")
        print(f"Original:   {original_text}")
        print(f"Anonymized: {anonymized_text_result}")

        # Check 1: The anonymized text should NOT be the same as the original.
        self.assertNotEqual(original_text, anonymized_text_result)

        # Check 2: None of the original PII values should appear in the final text.
        for pii_value in original_pii_values:
            self.assertNotIn(pii_value, anonymized_text_result)

if __name__ == '__main__':
    unittest.main()
