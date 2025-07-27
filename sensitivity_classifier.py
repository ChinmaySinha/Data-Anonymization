from transformers import pipeline

# Initialize the zero-shot classification pipeline once
try:
    classifier_pipeline = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli"
    )
except Exception as e:
    print(f"Error loading classifier model: {e}")
    classifier_pipeline = None

def classify_entity_sensitivity(entity_text: str) -> str:
    """
    Classifies a given entity's text into sensitivity levels using a zero-shot model.

    Args:
        entity_text: The text of the entity to classify (e.g., "John Doe").

    Returns:
        The sensitivity label with the highest score ('High', 'Medium', or 'Low').
    """
    if not classifier_pipeline:
        print("Classifier pipeline not available.")
        return "Unknown"

    # These labels can be customized based on your project's needs
    sensitivity_labels = ["High Sensitivity", "Medium Sensitivity", "Low Sensitivity"]
    
    result = classifier_pipeline(entity_text, candidate_labels=sensitivity_labels)
    
    # Return the label with the highest score
    return result['labels'][0]