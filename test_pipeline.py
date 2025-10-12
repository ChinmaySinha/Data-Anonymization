import unittest
from detectors import detect_all_pii
from sensitivity_classifier import classify_entity_sensitivity
from anonymization_module import anonymize_text

class TestAnonymizationPipeline(unittest.TestCase):

    def test_full_pipeline_with_presidio(self):
        """
        Tests the full PII anonymization pipeline with the new Presidio-based components.
        """
        # Sample text containing various PII entities, including our custom one
        original_text = "Contact Mr. John Doe at john.doe@example.com. His Polish ID is ABC123456 and his location is New York."

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

        # --- Step 4: Verification ---
        # The expected output reflects that "Polish" is detected as a low-sensitivity
        # entity (NRP) and masked, while the ID number is replaced as high-sensitivity.
        expected_anonymized_text = "Contact Mr. <PERSON> at <EMAIL_ADDRESS>. His ****** ID is <POLISH_IDENTITY_CARD> and his location is <LOCATION>."

        # We need to sort the entities by start position to apply replacements correctly
        # The anonymization module already handles this, so we can directly compare.

        print("\n--- Test Pipeline Results ---")
        print(f"Original: {original_text}")
        print(f"Anonymized: {anonymized_text_result}")
        print(f"Expected:   {expected_anonymized_text}")

        self.assertEqual(anonymized_text_result, expected_anonymized_text)

if __name__ == '__main__':
    unittest.main()
