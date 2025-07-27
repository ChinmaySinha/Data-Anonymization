import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    # --- THIS IS THE ONLY CHANGE ---
    # Point to the local folder containing your new, fine-tuned model.
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

    # 1. Tokenize input and get character-to-token mappings (offsets)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0]

    # 2. Run the model (inference)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 3. Get the predicted class for each token
    predictions = torch.argmax(logits, dim=2)[0]

    # 4. Reconstruct entities from tokens
    entities = []
    current_entity_tokens = []
    
    for i, token_prediction_id in enumerate(predictions):
        label_name = model.config.id2label[token_prediction_id.item()]
        
        if label_name != 'O':
            current_entity_tokens.append(i)
        
        is_end_of_entity = (label_name == 'O' or label_name.startswith('B-')) and current_entity_tokens
        is_end_of_text = (i == len(predictions) - 1) and current_entity_tokens

        if is_end_of_entity or is_end_of_text:
            start_char = offset_mapping[current_entity_tokens[0]][0].item()
            end_char = offset_mapping[current_entity_tokens[-1]][1].item()
            
            entity_group = model.config.id2label[predictions[current_entity_tokens[0]].item()].split('-')[-1]
            
            entities.append({
                'entity_group': entity_group,
                'word': text[start_char:end_char],
                'start': start_char,
                'end': end_char
            })
            # Reset for the next entity
            current_entity_tokens = []
            
            # If the current token itself is the start of a new entity, begin tracking it
            if label_name.startswith('B-'):
                current_entity_tokens.append(i)
                
    return entities