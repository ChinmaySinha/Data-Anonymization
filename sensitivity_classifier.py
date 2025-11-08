# Mapping of Presidio entity types to sensitivity levels
SENSITIVITY_MAP = {
    # High sensitivity entities
    "CREDIT_CARD": "High Sensitivity",
    "CRYPTO": "High Sensitivity",
    "IBAN_CODE": "High Sensitivity",
    "IP_ADDRESS": "High Sensitivity",
    "MEDICAL_LICENSE": "High Sensitivity",
    "PHONE_NUMBER": "High Sensitivity",
    "US_BANK_NUMBER": "High Sensitivity",
    "US_DRIVER_LICENSE": "High Sensitivity",
    "US_ITIN": "High Sensitivity",
    "US_PASSPORT": "High Sensitivity",
    "US_SSN": "High Sensitivity",
    "POLISH_IDENTITY_CARD": "High Sensitivity",

    # Medium sensitivity entities
    "PERSON": "Medium Sensitivity",
    "EMAIL_ADDRESS": "Medium Sensitivity",
    "DATE_TIME": "Medium Sensitivity",
    "LOCATION": "Medium Sensitivity",
    "URL": "Medium Sensitivity",

    # Low sensitivity entities
    "NRP": "Low Sensitivity",
    "ORGANIZATION": "Low Sensitivity",
}

def classify_entity_sensitivity(entity_group: str) -> str:
    """
    Classifies an entity's sensitivity based on its type using a predefined map.
    """
    # Look up the entity type in the map, defaulting to "Low" if not found
    return SENSITIVITY_MAP.get(entity_group, "Low Sensitivity")
