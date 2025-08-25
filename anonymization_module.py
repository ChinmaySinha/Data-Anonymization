from faker import Faker

# Initialize Faker
faker = Faker('en_IN')

# Mapping from our model's entity types to the correct Faker function
FAKER_PROVIDER_MAP = {
    'PERSON': faker.name,
    'PER': faker.name,
    'NAME': faker.name,
    'CITY': faker.city,
    'LOCATION': faker.address,
    'ORGANIZATION': faker.company,
    'ORG': faker.company,
    'USERNAME': faker.user_name,
    'EMAIL': faker.email,
    'PHONE_NUMBER': faker.phone_number,
    'TEL': faker.phone_number,
    'IP_ADDRESS': faker.ipv4,
    'IP': faker.ipv4,
    'ID_NUM': faker.ssn,
}

def pseudonymize(entity_text, entity_type, pseudonym_map):
    """
    Replaces an entity with a consistent but realistic fake value.
    """
    if entity_text not in pseudonym_map:
        provider = FAKER_PROVIDER_MAP.get(entity_type.upper())
        if provider:
            pseudonym_map[entity_text] = provider()
        else:
            new_id = len(pseudonym_map) + 1
            pseudonym_map[entity_text] = f"[{entity_type.upper()}_{new_id}]"
    
    return pseudonym_map[entity_text]

def generalize(entity_type):
    """
    Replaces an entity with its general category type (e.g., [CITY]).
    """
    return f"[{entity_type.upper()}]"

def anonymize_text(original_text, entities_with_sensitivity):
    """
    Applies anonymization techniques based on a 3-tiered sensitivity threshold.
    """
    anonymized_text = original_text
    pseudonym_map = {}
    
    for entity in sorted(entities_with_sensitivity, key=lambda x: x['start'], reverse=True):
        start = entity['start']
        end = entity['end']
        entity_type = entity['entity_group']
        sensitivity = entity['sensitivity']
        original_entity_text = entity['word']

        replacement_text = ""

        # --- FINAL 3-TIERED THRESHOLD LOGIC ---
        if sensitivity == "High Sensitivity":
            # High sensitivity -> Pseudonymize with realistic data
            replacement_text = pseudonymize(original_entity_text, entity_type, pseudonym_map)
        
        elif sensitivity == "Medium Sensitivity":
            # Medium sensitivity -> Generalize with a category tag
            replacement_text = generalize(entity_type)
        
        # If sensitivity is "Low Sensitivity", we do nothing.
        # replacement_text remains "" and the text is not changed.
        # ---------------------------------------------------

        if replacement_text:
            anonymized_text = anonymized_text[:start] + replacement_text + anonymized_text[end:]
            
    return anonymized_text
