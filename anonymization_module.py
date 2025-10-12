from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize the AnonymizerEngine
anonymizer = AnonymizerEngine()

def anonymize_text(original_text: str, entities_with_sensitivity: list) -> str:
    """
    Anonymizes text using Presidio's AnonymizerEngine based on sensitivity.
    """
    # Define operators for different sensitivity levels
    operators = {}
    for entity in entities_with_sensitivity:
        sensitivity = entity['sensitivity']
        entity_type = entity['entity_group']

        if sensitivity == "High Sensitivity" or sensitivity == "Medium Sensitivity":
            # For High/Medium sensitivity, replace with a placeholder like <PERSON>
            operators[entity_type] = OperatorConfig("replace", {"new_value": f"<{entity_type}>"})
        elif sensitivity == "Low Sensitivity":
            # For Low sensitivity, mask with a fixed character
            operators[entity_type] = OperatorConfig("mask", {"type": "fixed", "masking_char": "*", "chars_to_mask": len(entity['word']), "from_end": False})

    # Convert our entity format to Presidio's AnalyzerResult format
    analyzer_results = []
    for entity in entities_with_sensitivity:
        from presidio_analyzer import RecognizerResult
        analyzer_results.append(RecognizerResult(
            entity_type=entity['entity_group'],
            start=entity['start'],
            end=entity['end'],
            score=entity.get('score', 0.85)  # Use detected score or a default
        ))

    # Anonymize the text
    anonymized_result = anonymizer.anonymize(
        text=original_text,
        analyzer_results=analyzer_results,
        operators=operators
    )

    return anonymized_result.text
