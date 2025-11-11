import spacy
from ner_detector import detect_entities_with_ner
from regex_matcher import detect_pii_with_regex
from presidio_detector import detect_pii_with_presidio

# --- Load the SpaCy model ---
try:
    spacy_nlp = spacy.load("en_core_web_lg")
    # print("Successfully loaded the SpaCy model for the expert committee.")
except Exception as e:
    print(f"Could not load the SpaCy model. This part of the detector will be skipped. Did you run 'python -m spacy download en_core_web_lg'? Error: {e}")
    spacy_nlp = None

def _detect_pii_with_spacy(text: str) -> list:
    """Helper function to detect PII with the spaCy model."""
    if not spacy_nlp:
        return []
    doc = spacy_nlp(text)
    spacy_entities = []
    for ent in doc.ents:
        label = "LOCATION" if ent.label_ == "GPE" else ent.label_
        spacy_entities.append({'entity_group': label, 'word': ent.text, 'start': ent.start_char, 'end': ent.end_char})
    return spacy_entities

def detect_all_pii(text: str) -> list:
    """
    Runs a hybrid of all available detectors in a priority order
    and returns a combined, de-duplicated list of PII.
    """
    
    # We will build the final list and keep track of covered indices
    final_detections = []
    covered_indices = set()

    # --- 1. Run Presidio First (Highest Priority) ---
    # Presidio is good at general, high-confidence entities like ORG, LOC, etc.
    presidio_results = detect_pii_with_presidio(text)
    for entity in presidio_results:
        # Check for overlap: if not any(i in covered_indices for i in range(entity['start'], entity['end'])):
        if all(i not in covered_indices for i in range(entity['start'], entity['end'])):
            final_detections.append(entity)
            for i in range(entity['start'], entity['end']):
                covered_indices.add(i)

    # --- 2. Run Fine-Tuned NER (Second Priority) ---
    # Our model is good at specific, trained entities (like USERNAME)
    ner_results = detect_entities_with_ner(text)
    for entity in ner_results:
        if all(i not in covered_indices for i in range(entity['start'], entity['end'])):
            final_detections.append(entity)
            for i in range(entity['start'], entity['end']):
                covered_indices.add(i)

    # --- 3. Run SpaCy (Third Priority) ---
    # spaCy is a general-purpose model, we use it to fill any remaining gaps
    spacy_results = _detect_pii_with_spacy(text)
    for entity in spacy_results:
        if all(i not in covered_indices for i in range(entity['start'], entity['end'])):
            final_detections.append(entity)
            for i in range(entity['start'], entity['end']):
                covered_indices.add(i)

    # --- 4. Run Regex (Fourth Priority) ---
    # Regex is for very specific patterns (like IP) that might have been missed
    regex_results = detect_pii_with_regex(text)
    for entity in regex_results:
        if all(i not in covered_indices for i in range(entity['start'], entity['end'])):
            final_detections.append(entity)
            for i in range(entity['start'], entity['end']):
                covered_indices.add(i)

    # Sort the final list by start position for clean processing downstream
    final_detections = sorted(final_detections, key=lambda x: x['start'])

    return final_detections