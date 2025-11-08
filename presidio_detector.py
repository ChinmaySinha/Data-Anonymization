from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Define a custom recognizer for Polish Identity Card numbers
# This is a simple regex for the format 'AAA123456'
polish_id_pattern = Pattern(name="polish_id_pattern", regex=r'[A-Z]{3}\d{6}', score=0.8)
polish_id_recognizer = PatternRecognizer(supported_entity="POLISH_IDENTITY_CARD", patterns=[polish_id_pattern])

# Initialize the AnalyzerEngine with the custom recognizer
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(polish_id_recognizer)

def detect_pii_with_presidio(text: str) -> list:
    """
    Detects PII entities using the Presidio Analyzer.
    """
    results = analyzer.analyze(text=text, language='en')

    # Convert Presidio's results to the format expected by our pipeline
    detected_entities = []
    for res in results:
        detected_entities.append({
            'entity_group': res.entity_type,
            'word': text[res.start:res.end],
            'start': res.start,
            'end': res.end,
            'score': res.score
        })

    return detected_entities

if __name__ == '__main__':
    # Example usage
    sample_text = "My name is John Doe, and my Polish ID is ABC123456. My email is john.doe@example.com."

    print("--- Detecting PII with Presidio ---")
    pii_results = detect_pii_with_presidio(sample_text)

    if pii_results:
        for pii in pii_results:
            print(f"  - Entity: '{pii['word']}' ({pii['entity_group']}), Score: {pii['score']:.2f}")
    else:
        print("No PII detected.")
