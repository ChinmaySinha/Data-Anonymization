from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import RecognizerResult
from faker import Faker

# Initialize the AnonymizerEngine and Faker
anonymizer = AnonymizerEngine()
faker = Faker("en_IN") # Use Indian locale for context-appropriate fake data

# Mapping of Presidio entity types to Faker method names
FAKER_PROVIDER_MAP = {
    'PERSON': 'name',
    'EMAIL_ADDRESS': 'email',
    'PHONE_NUMBER': 'phone_number',
    'LOCATION': 'address',
    'URL': 'url',
    'IP_ADDRESS': 'ipv4',
    'CREDIT_CARD': 'credit_card_number',
    'US_SSN': 'ssn',
    'US_DRIVER_LICENSE': 'license_plate',
    'DATE_TIME': 'date',
    'POLISH_IDENTITY_CARD': 'ssn',
    'USERNAME': 'name', # <-- This is your change
}

def anonymize_text(original_text: str, entities_with_sensitivity: list) -> str:
    """
    Anonymizes text using Presidio's AnonymizerEngine, replacing PII with
    realistic fake data using a custom Faker operator.
    """
    operators = {}
    for entity in entities_with_sensitivity:
        sensitivity = entity['sensitivity']
        entity_type = entity['entity_group']

        if sensitivity == "High Sensitivity" or sensitivity == "Medium Sensitivity":
            faker_provider_name = FAKER_PROVIDER_MAP.get(entity_type)
            if faker_provider_name:
                # The 'lambda' parameter expects a callable function.
                # We dynamically get the correct Faker method (e.g., faker.name)
                # and create a lambda that calls it.
                faker_method = getattr(faker, faker_provider_name)
                
                # --- THIS IS THE CORRECTED LINE ---
                operators[entity_type] = OperatorConfig(
                    "custom",
                    {"lambda": lambda x: faker_method()}
                )
            else:
                operators[entity_type] = OperatorConfig("replace", {"new_value": f"<{entity_type}>"})

        elif sensitivity == "Low Sensitivity":
            operators[entity_type] = OperatorConfig(
                "mask",
                {"type": "fixed", "masking_char": "*", "chars_to_mask": len(entity['word']), "from_end": False}
            )

    analyzer_results = []
    for entity in entities_with_sensitivity:
        analyzer_results.append(RecognizerResult(
            entity_type=entity['entity_group'],
            start=entity['start'],
            end=entity['end'],
            score=entity.get('score', 0.85)
        ))

    anonymized_result = anonymizer.anonymize(
        text=original_text,
        analyzer_results=analyzer_results,
        operators=operators
    )

    return anonymized_result.text