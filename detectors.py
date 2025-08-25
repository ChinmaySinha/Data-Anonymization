import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy

# --- Detector 1: Your Fine-Tuned "Super-Model" ---
try:
    finetuned_model_name = "./roberta-large-pii-finetuned" 
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
    finetuned_model = AutoModelForTokenClassification.from_pretrained(finetuned_model_name)
    print("✅ Successfully loaded the fine-tuned super-model.")
except Exception as e:
    print(f"❌ Error loading fine-tuned model: {e}")
    finetuned_model = None

# --- Detector 2: The Fast, General Expert (SpaCy) ---
try:
    spacy_nlp = spacy.load("en_core_web_lg")
    print("✅ Successfully loaded the SpaCy model.")
except Exception as e:
    print(f"❌ Error loading SpaCy model. Did you run 'python -m spacy download en_core_web_lg'? Error: {e}")
    spacy_nlp = None

# --- Detector 3: The Regex Matcher ---
def detect_pii_with_regex(text: str) -> list:
    patterns = {
        'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'PHONE_NUMBER': r'\+91-?\d{10}',
        'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }
    found_pii = []
    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            found_pii.append({'entity_group': pii_type, 'word': match.group(0), 'start': match.start(), 'end': match.end()})
    return found_pii

# Helper function for your fine-tuned model
def _run_transformer_ner(text, model, tokenizer):
    if not model or not tokenizer: return []
    # (This function's internal logic is unchanged)
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    entities = []
    current_entity = None
    for i, token_id in enumerate(predictions):
        label = model.config.id2label[token_id]
        if offset_mapping[i] == [0, 0]: continue
        if label.startswith('B-'):
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
            current_entity = {"entity_group": label[2:], "start": offset_mapping[i][0], "end": offset_mapping[i][1]}
        elif label.startswith('I-') and current_entity and label[2:] == current_entity["entity_group"]:
            current_entity["end"] = offset_mapping[i][1]
        else:
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        current_entity['word'] = text[current_entity['start']:current_entity['end']]
        entities.append(current_entity)
    return entities

# New helper function for the SpaCy model
def detect_pii_with_spacy(text: str) -> list:
    if not spacy_nlp: return []
    doc = spacy_nlp(text)
    spacy_entities = []
    for ent in doc.ents:
        # SpaCy uses 'GPE' for geopolitical entities like cities/countries
        label = "LOCATION" if ent.label_ == "GPE" else ent.label_
        spacy_entities.append({'entity_group': label, 'word': ent.text, 'start': ent.start_char, 'end': ent.end_char})
    return spacy_entities

def detect_all_pii(text: str) -> list:
    """
    Runs all available detectors and returns a combined, de-duplicated list of PII.
    """
    print("\n--- 1. Detecting PII entities with the expert team... ---")
    
    # Run all three detectors
    finetuned_results = _run_transformer_ner(text, finetuned_model, finetuned_tokenizer)
    print(f"  - Fine-Tuned Model found {len(finetuned_results)} items.")
    spacy_results = detect_pii_with_spacy(text)
    print(f"  - SpaCy Model found {len(spacy_results)} items.")
    regex_results = detect_pii_with_regex(text)
    print(f"  - Regex Matcher found {len(regex_results)} items.")
    
    # Combine all results
    all_detections = finetuned_results + spacy_results + regex_results
    
    # Remove duplicate/overlapping entities
    unique_detections = []
    seen_positions = set()
    for item in sorted(all_detections, key=lambda x: x['start']):
        pos = (item['start'], item['end'])
        if pos in seen_positions: continue
        is_subset = any(p_start <= item['start'] and p_end >= item['end'] for p_start, p_end in seen_positions)
        if not is_subset:
            unique_detections.append(item)
            seen_positions.add(pos)
            
    print(f"Total unique entities detected: {len(unique_detections)}.")
    return unique_detections
