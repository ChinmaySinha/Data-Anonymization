import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    # Point to the local folder containing your fine-tuned model
    model_name = "./roberta-large-pii-finetuned"

    # CORRECTED: Added 'local_files_only=True' to prevent remote lookup errors
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
    print(f"Successfully loaded fine-tuned model from '{model_name}' for expert committee.")

except Exception as e:
    print(f"Could not load the fine-tuned model for the expert committee. This part of the detector will be skipped. Error: {e}")
    model = None
    tokenizer = None

def detect_entities_with_ner(text: str) -> list:
    """
    Detects PII entities using the fine-tuned transformer model.
    """
    if not model or not tokenizer:
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

    entities = []
    current_entity = None

    for i, token_prediction_id in enumerate(predictions):
        label_name = model.config.id2label[token_prediction_id]

        if offset_mapping[i] == [0, 0]:
            continue

        if label_name.startswith('B-'):
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)

            current_entity = {
                "entity_group": label_name[2:],
                "start": offset_mapping[i][0],
                "end": offset_mapping[i][1]
            }
        elif label_name.startswith('I-') and current_entity:
            if label_name[2:] == current_entity["entity_group"]:
                current_entity["end"] = offset_mapping[i][1]
            else:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = {
                    "entity_group": label_name[2:],
                    "start": offset_mapping[i][0],
                    "end": offset_mapping[i][1]
                }
        else:
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        current_entity['word'] = text[current_entity['start']:current_entity['end']]
        entities.append(current_entity)

    return entities
