from presidio_detector import detect_pii_with_presidio

def detect_all_pii(text: str) -> list:
    """
    Detects all PII entities using the Presidio-based detector.
    This function serves as a single, unified entry point for PII detection.
    """
    print("\n--- 1. Detecting PII entities with Presidio... ---")
    
    # Call the new Presidio detector function
    detected_entities = detect_pii_with_presidio(text)
    
    print(f"Total entities detected by Presidio: {len(detected_entities)}.")
    
    return detected_entities
