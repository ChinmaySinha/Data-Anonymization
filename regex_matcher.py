import re

def detect_pii_with_regex(text: str) -> list:
    """
    Detects fixed-format PII using regular expressions.
    """
    # Patterns for common PII types
    patterns = {
        'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'PHONE_NUMBER': r'\+91-?\d{10}',
        'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }

    found_pii = []
    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            found_pii.append({
                'entity_group': pii_type,
                'word': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    return found_pii
