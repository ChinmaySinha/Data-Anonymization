import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    # Point to the local folder containing your fine-tuned model
    model_name = "./roberta-base-pii-finetuned"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    print(f"Successfully loaded fine-tuned model from '{model_name}'")

except Exception as e:
    print(f"Error loading NER model: {e}")
    model = None
    tokenizer = None

def detect_entities_with_ner(text: str) -> list:
    """
    Detects PII entities using a transformer model directly.
    """
    if not model or not tokenizer:
        print("NER model/tokenizer not available.")
        return []

    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=2)[0].tolist()
    
    # --- CORRECTED ENTITY RECONSTRUCTION LOGIC ---
    entities = []
    current_entity = None

    for i, token_prediction_id in enumerate(predictions):
        label_name = model.config.id2label[token_prediction_id]
        
        # Skip special tokens
        if offset_mapping[i] == [0, 0]:
            continue

        if label_name.startswith('B-'):
            # If we have a current entity, save it before starting a new one
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
            
            # Start a new entity
            current_entity = {
                "entity_group": label_name[2:],
                "start": offset_mapping[i][0],
                "end": offset_mapping[i][1]
            }
        elif label_name.startswith('I-') and current_entity:
            # If it's a continuation of the same entity type, extend it
            if label_name[2:] == current_entity["entity_group"]:
                current_entity["end"] = offset_mapping[i][1]
            else:
                # If it's a different entity type, save the old one and start a new one
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = {
                    "entity_group": label_name[2:],
                    "start": offset_mapping[i][0],
                    "end": offset_mapping[i][1]
                }
        else:
            # If it's 'O' or an unexpected tag, save the current entity and reset
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = None
    
    # Add the last entity if the text ends with it
    if current_entity:
        current_entity['word'] = text[current_entity['start']:current_entity['end']]
        entities.append(current_entity)

    return entities