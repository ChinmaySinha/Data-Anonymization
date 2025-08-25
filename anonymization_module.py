from faker import Faker

# Initialize Faker. We can specify 'en_IN' for Indian-context names.
faker = Faker('en_IN')

# Create a mapping from our model's entity types to the correct Faker function
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
        # Check if we have a specific Faker provider for this entity type
        provider = FAKER_PROVIDER_MAP.get(entity_type.upper())
        if provider:
            # Generate a new, realistic fake value
            pseudonym_map[entity_text] = provider()
        else:
            # Fallback for unknown entity types
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
    Applies advanced anonymization techniques based on entity sensitivity.
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

        # --- FINAL CORRECTED THRESHOLD LOGIC ---
        if sensitivity == "High Sensitivity" or sensitivity == "Medium Sensitivity":
            # For High and Medium sensitivity, we will pseudonymize with realistic data.
            replacement_text = pseudonymize(original_entity_text, entity_type, pseudonym_map)
        
        elif sensitivity == "Low Sensitivity":
            # For Low sensitivity, we will generalize with a category tag.
            replacement_text = generalize(entity_type)
        # ----------------------------------------

        if replacement_text:
            anonymized_text = anonymized_text[:start] + replacement_text + anonymized_text[end:]
            
    return anonymized_text
