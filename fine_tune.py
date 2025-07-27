import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("─" * 80)
    print(f"🚀 Training will run on: {device.upper()}")
    print("─" * 80)

    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "pii-masking-300k")
    dataset_dict = load_dataset(local_dataset_path)
    
    unique_labels = set(label['label'] for split in dataset_dict.values() for item in split for label in item['privacy_mask'])
    label_list = sorted(list(unique_labels))
    bio_label_list = ["O"] + [f"{tag}-{label}" for label in label_list for tag in ["B", "I"]]
    bio_id2label = {i: label for i, label in enumerate(bio_label_list)}
    bio_label2id = {label: i for i, label in enumerate(bio_label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(bio_label_list),
        id2label=bio_id2label,
        label2id=bio_label2id
    )

    def align_labels_with_tokens(example):
        tokenized_inputs = tokenizer(example["source_text"], truncation=True, is_split_into_words=False)
        labels = []
        word_ids = tokenized_inputs.word_ids()
        
        char_to_label = ['O'] * len(example["source_text"])
        for pii in example["privacy_mask"]:
            pii_type = pii["label"]
            start, end = pii["start"], pii["end"]
            if start < len(char_to_label):
                char_to_label[start] = f"B-{pii_type}"
                for i in range(start + 1, end):
                    if i < len(char_to_label):
                        char_to_label[i] = f"I-{pii_type}"

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                try:
                    start_char_pos = tokenized_inputs.word_to_chars(word_idx).start
                    label = char_to_label[start_char_pos]
                    if previous_word_idx == word_idx and label.startswith("B-"):
                        label = "I-" + label[2:]
                    labels.append(bio_label2id[label])
                except (KeyError, IndexError):
                    labels.append(bio_label2id['O'])
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset_dict.map(align_labels_with_tokens, batched=False)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[bio_label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[bio_label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
        
    output_dir = "roberta-base-pii-finetuned"
    
    # --- THIS SECTION IS NOW CORRECTED ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="no",      # Corrected argument name
        save_strategy="epoch",
        fp16=True,
        max_steps=50
    )

    # Create a smaller subset for evaluation
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"].select(range(1000))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- RUN TRAINING AND EVALUATION SEPARATELY ---
    print("--- Starting Fine-Tuning (Training Only) ---")
    trainer.train()
    
    print("--- Saving model ---")
    trainer.save_model(output_dir)
    
    print("--- Starting Evaluation ---")
    metrics = trainer.evaluate()
    
    print("--- Fine-Tuning and Evaluation Complete ---")
    print("Evaluation Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()