# Advanced PII Anonymization Pipeline

This project provides a sophisticated pipeline for detecting and anonymizing Personally Identifiable Information (PII) in text. It uses a multi-layered approach to ensure high accuracy and context-aware anonymization, making it suitable for a wide range of privacy-sensitive applications.

## Key Features

*   **Hybrid PII Detection:** Employs an "expert committee" of three different detectors to maximize PII detection accuracy:
    *   A fine-tuned RoBERTa-large model for state-of-the-art Named Entity Recognition (NER).
    *   A spaCy model for fast and general-purpose entity detection.
    *   Regex patterns for specific PII formats like emails, phone numbers, and IP addresses.
*   **Sensitivity-Based Anonymization:** Goes beyond simple PII masking by first classifying the sensitivity of each detected entity (High, Medium, or Low).
*   **Intelligent Anonymization Techniques:**
    *   **Pseudonymization:** Replaces High and Medium sensitivity PII with realistic, context-appropriate fake data (e.g., "John Doe" becomes "Aarav Sharma").
    *   **Generalization:** Replaces Low sensitivity PII with its general category (e.g., "New Delhi" becomes "[CITY]").
*   **Extensible and Modular:** The codebase is organized into logical modules for detection, classification, and anonymization, making it easy to extend or customize.
*   **Performance Evaluation:** Includes a script to evaluate the PII detection performance using standard metrics like precision, recall, and F1-score.

## How It Works

The pipeline processes text in a series of steps:

1.  **PII Detection:** The input text is fed through the `detectors.py` module, which runs all three PII detectors (fine-tuned transformer, spaCy, and regex) and combines their findings into a single, de-duplicated list of PII entities.
2.  **Sensitivity Classification:** Each detected PII entity is classified as "High," "Medium," or "Low" sensitivity by the `sensitivity_classifier.py` module, which uses a zero-shot classification model.
3.  **Anonymization:** The `anonymization_module.py` module processes the original text and the list of classified PII entities. Based on the sensitivity level, it replaces each entity with either a realistic pseudonym or a general category tag.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2.  **Install the required Python packages:**
    ```bash
    pip install -r requirement.txt
    ```
3.  **Download the spaCy model:**
    ```bash
    python -m spacy download en_core_web_lg
    ```
4.  **Download the dataset (for fine-tuning):**
    The project uses the `ai4privacy/pii-masking-300k` dataset. The `fine_tune.py` script will automatically download it, but you can also clone it manually from the Hugging Face Hub:
    ```bash
    git clone https://huggingface.co/datasets/ai4privacy/pii-masking-300k
    ```

## Usage

There are two main ways to use this project:

### 1. Anonymize Text

To anonymize a piece of text, run the `main.py` script. You can either provide your own text as input or use a default sample from the dataset.

```bash
python main.py
```

The script will then guide you through the process and display the original and anonymized text.

### 2. Fine-Tune the Model

To fine-tune the RoBERTa model on the PII dataset, run the `fine_tune.py` script. This will train the model and save the fine-tuned version to the `roberta-large-pii-finetuned` directory.

```bash
python fine_tune.py
```

**Note:** Fine-tuning is a resource-intensive process and may take a significant amount of time and computational power.

## Performance

The PII detection pipeline achieves the following performance metrics on the evaluation dataset:

*   **Precision:** 0.9718
*   **Recall:** 0.9766
*   **F1-Score:** 0.9742

## File Descriptions

*   `main.py`: The main entry point for the anonymization pipeline.
*   `anonymization_module.py`: Handles the anonymization of text based on sensitivity levels.
*   `detectors.py`: Combines multiple PII detection methods (transformer, spaCy, regex).
*   `ner_detector.py`: A standalone PII detector using a fine-tuned transformer model.
*   `regex_matcher.py`: A standalone PII detector using regular expressions.
*   `sensitivity_classifier.py`: Classifies the sensitivity of detected PII entities.
*   `pii_evaluate.py`: Calculates performance metrics for PII detection.
*   `fine_tune.py`: The script for fine-tuning the RoBERTa model.
*   `requirement.txt`: A list of the required Python packages for the project.

## Future works:
  To make this system contect-aware using LLMs (working on that)
