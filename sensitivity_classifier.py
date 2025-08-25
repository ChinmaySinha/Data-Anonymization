from transformers import pipeline

try:
    # Initialize the zero-shot classification pipeline once for efficiency
    # This model is good at classifying text without specific training
    classifier_pipeline = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli"
    )
    print("Sensitivity classification model loaded successfully.")
except Exception as e:
    print(f"Error loading classifier model: {e}")
    classifier_pipeline = None

def classify_entity_sensitivity(entity_text: str) -> str:
    """
    Classifies a given entity's text into sensitivity levels.

    Args:
        entity_text: The text of the entity to classify (e.g., "John Doe").

    Returns:
        The sensitivity label with the highest score ('High', 'Medium', or 'Low').
    """
    if not classifier_pipeline:
        print("Classifier pipeline not available.")
        return "Unknown"

    # Define the labels for our classification
    sensitivity_labels = ["High Sensitivity", "Medium Sensitivity", "Low Sensitivity"]
    
    # The model returns scores for each label; we pick the highest one
    result = classifier_pipeline(entity_text, candidate_labels=sensitivity_labels)
    
    return result['labels'][0]
