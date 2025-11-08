# High-Performance Hybrid PII Anonymization Pipeline

This project provides a sophisticated and high-performance pipeline for detecting and anonymizing Personally Identifiable Information (PII) in text. It uses a **hybrid detection engine** that combines a fine-tuned transformer model, spaCy, regex, and the Presidio library to ensure maximum accuracy.

## Key Features

*   **State-of-the-Art Hybrid PII Detection:** Employs a powerful hybrid engine to maximize PII detection accuracy:
    *   A fine-tuned RoBERTa-large model for high-precision Named Entity Recognition (NER).
    *   The **Presidio Analyzer** for broad-coverage PII detection.
    *   A spaCy model for fast, general-purpose entity detection.
    *   Custom Regex patterns for specific PII formats (e.g., emails, phone numbers).
*   **High-Performance Sensitivity Classification:** Uses a fast, rule-based approach to classify the sensitivity of each detected entity (High, Medium, or Low).
*   **Intelligent Anonymization with Faker:**
    *   **Pseudonymization:** Replaces High and Medium sensitivity PII with realistic, context-appropriate fake data using the `Faker` library.
    *   **Masking:** Replaces Low sensitivity PII with a character mask.
*   **Performance Evaluation:** Includes a dedicated script to evaluate the PII detection performance of the hybrid engine, providing key metrics like Precision, Recall, and F1-Score.

## How It Works

The pipeline processes text in a series of steps:

1.  **Hybrid PII Detection:** The input text is fed through the `detectors.py` module, which runs all detection methods (Presidio, fine-tuned transformer, spaCy, and regex) and combines their findings into a single, de-duplicated list of PII entities.
2.  **Sensitivity Classification:** Each detected entity's type is classified as "High," "Medium," or "Low" sensitivity by the high-speed, rule-based classifier.
3.  **Anonymization:** Based on the sensitivity level, the Presidio Anonymizer replaces each entity with either realistic fake data from `Faker` or a character mask.

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
4.  **Provide Local Models & Data:**
    *   **Fine-tuned Model:** Ensure your fine-tuned transformer model is located in a directory named `roberta-large-pii-finetuned` in the project root.
    *   **Dataset:** For performance evaluation, ensure the `pii-masking-300k` dataset is located in the project root.

## Usage

### Anonymize Text
To anonymize a piece of text, run the `main.py` script.
```bash
python main.py
```

### Evaluate Performance
To evaluate the performance of the hybrid detection engine, run the `evaluate_pipeline.py` script. This requires the `pii-masking-300k` dataset to be present.
```bash
python evaluate_pipeline.py
```

## File Descriptions

*   `main.py`: The main entry point for the anonymization pipeline.
*   `detectors.py`: Implements the **hybrid detection engine**, combining all PII detection methods.
*   `presidio_detector.py`: A module for PII detection using the **Presidio Analyzer**.
*   `ner_detector.py`: A standalone PII detector using a fine-tuned transformer model.
*   `regex_matcher.py`: A standalone PII detector using regular expressions.
*   `sensitivity_classifier.py`: A high-performance, rule-based classifier for determining PII sensitivity.
*   `anonymization_module.py`: Handles the anonymization of text using the **Presidio Anonymizer** and `Faker`.
*   `evaluate_pipeline.py`: A script to evaluate the performance of the detection pipeline.
*   `test_pipeline.py`: An end-to-end test for the full anonymization pipeline.
*   `requirement.txt`: A list of the required Python packages for the project.

## Future works:
  To make this system contect-aware using LLMs (working on that)
