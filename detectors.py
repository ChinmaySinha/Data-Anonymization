import spacy
from ner_detector import detect_entities_with_ner
from regex_matcher import detect_pii_with_regex
from presidio_detector import detect_pii_with_presidio

# --- Load the SpaCy model (part of the expert committee) ---
try:
    spacy_nlp = spacy.load("en_core_web_lg")
    print("Successfully loaded the SpaCy model for the expert committee.")
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
    Runs a hybrid of all available detectors (Presidio and the expert committee)
    and returns a combined, de-duplicated list of PII.
    """
    print("\n--- 1. Detecting PII entities with the HYBRID engine... ---")

    # --- Run all detectors ---
    presidio_results = detect_pii_with_presidio(text)
    print(f"  - Presidio Analyzer found {len(presidio_results)} items.")

    ner_results = detect_entities_with_ner(text)
    print(f"  - Fine-Tuned NER Model found {len(ner_results)} items.")
    
    spacy_results = _detect_pii_with_spacy(text)
    print(f"  - SpaCy Model found {len(spacy_results)} items.")

    regex_results = detect_pii_with_regex(text)
    print(f"  - Regex Matcher found {len(regex_results)} items.")
    
    # --- Combine and De-duplicate ---
    all_detections = presidio_results + ner_results + spacy_results + regex_results
    
    # Remove duplicate/overlapping entities.
    # We prioritize longer matches and give preference to the order of detection (Presidio first).
    unique_detections = []
    seen_positions = set()

    # Sort by start position, then by length (longest first) to prioritize more specific matches
    sorted_detections = sorted(all_detections, key=lambda x: (x['start'], x['end'] - x['start']), reverse=True)

    final_detections = []
    covered_indices = set()

    for entity in sorted_detections:
        # Check if the entity's character indices overlap with already covered indices
        if any(i in covered_indices for i in range(entity['start'], entity['end'])):
            continue  # Skip this entity as it overlaps with a better one we've already chosen

        # If not overlapping, add it to our final list and mark its indices as covered
        final_detections.append(entity)
        for i in range(entity['start'], entity['end']):
            covered_indices.add(i)

    # Sort the final list by start position for clean processing downstream
    final_detections = sorted(final_detections, key=lambda x: x['start'])

    print(f"Total unique entities detected by hybrid engine: {len(final_detections)}.")
    return final_detections
