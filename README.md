# High-Performance PII Anonymization Pipeline with Presidio

This project provides a sophisticated and high-performance pipeline for detecting and anonymizing Personally Identifiable Information (PII) in text. It leverages the powerful **Presidio** library to ensure high accuracy and context-aware anonymization, making it suitable for a wide range of privacy-sensitive applications.

## Key Features

*   **Robust PII Detection with Presidio:** Uses the industry-standard Presidio library to accurately detect a wide range of PII entities, including names, locations, credit card numbers, and more.
*   **Custom PII Recognition:** The pipeline is extensible and includes a custom recognizer for "Polish Identity Card" as an example.
*   **High-Performance Sensitivity Classification:** Employs a fast, rule-based approach to classify the sensitivity of each detected entity (High, Medium, or Low) without the need for slow, resource-intensive AI models.
*   **Intelligent Anonymization with Faker:**
    *   **Pseudonymization:** Replaces High and Medium sensitivity PII with realistic, context-appropriate fake data using the `Faker` library (e.g., "John Doe" becomes "Aarav Sharma").
    *   **Masking:** Replaces Low sensitivity PII with a fixed character mask (e.g., "Polish" becomes "******").
*   **Extensible and Modular:** The codebase is organized into logical modules for detection, classification, and anonymization.
*   **End-to-End Testing:** Includes a test suite to validate the full functionality of the pipeline.

## How It Works

The streamlined pipeline processes text in a series of steps:

1.  **PII Detection:** The input text is fed into the **Presidio Analyzer**, which detects all known and custom PII entities.
2.  **Sensitivity Classification:** Each detected PII entity's type is passed to a high-speed, rule-based classifier that instantly determines its sensitivity level (High, Medium, or Low) based on a predefined map.
3.  **Anonymization:** Based on the sensitivity level, the **Presidio Anonymizer** replaces each entity with either realistic fake data from `Faker` or a character mask.

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
3.  **Download the spaCy model (used by Presidio):**
    ```bash
    python -m spacy download en_core_web_lg
    ```

## Usage

To anonymize a piece of text, run the `main.py` script. You can either provide your own text as input or use a default sample.

```bash
python main.py
```

The script will guide you through the process and display the original and anonymized text.

To run the test suite and verify the pipeline's functionality:
```bash
python test_pipeline.py
```

## File Descriptions

*   `main.py`: The main entry point for the anonymization pipeline.
*   `presidio_detector.py`: A dedicated module for PII detection using the **Presidio Analyzer**.
*   `sensitivity_classifier.py`: A high-performance, rule-based classifier for determining PII sensitivity.
*   `anonymization_module.py`: Handles the anonymization of text using the **Presidio Anonymizer** and `Faker`.
*   `test_pipeline.py`: An end-to-end test for the full anonymization pipeline.
*   `pii_evaluate.py`: Calculates performance metrics for PII detection.
*   `fine_tune.py`: An optional script for fine-tuning a transformer model (not used in the main pipeline).
*   `requirement.txt`: A list of the required Python packages for the project.

## Future works:
  To make this system contect-aware using LLMs (working on that)
