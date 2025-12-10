#!/usr/bin/env python3
"""
Text Classification following Hugging Face Transformers tutorial.
Based on: https://huggingface.co/docs/transformers/en/tasks/sequence_classification
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from huggingface_hub import login

# Optional: Login to Hugging Face to upload model
# login()

# Load dataset from CSV file
print("Loading dataset from CSV...")
dataset = load_dataset("csv", data_files="lign167_projectDataset/test_data.csv")
print(dataset)

# Split dataset into train and test sets
print("\nSplitting dataset into train and test...")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
print(dataset)

# Take a look at an example
print("\nExample from dataset:")
print(dataset["train"][0])

# Load DistilBERT tokenizer
print("\nLoading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Create a preprocessing function to tokenize text
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Apply preprocessing function to entire dataset
print("\nTokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load evaluation metric
print("\nLoading accuracy metric...")
accuracy = evaluate.load("accuracy")

# Create compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Create label mappings (0 = inaccurate, 1 = accurate)
id2label = {0: "INACCURATE", 1: "ACCURATE"}
label2id = {"INACCURATE": 0, "ACCURATE": 1}

# Load model
print("\nLoading DistilBERT model for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("\nStarting training...")
trainer.train()

# Save the model
print("\nSaving model...")
trainer.save_model("./my_awesome_model")

print("\nTraining complete!")

# Inference example
print("\n" + "="*70)
print("INFERENCE EXAMPLE")
print("="*70)

# Test with a sample text from the dataset
text = "CRISPR-Cas9 creates double-strand breaks in DNA through homologous recombination repair."

from transformers import pipeline

classifier = pipeline("text-classification", model="my_awesome_model", tokenizer=tokenizer)
result = classifier(text)
print(f"\nText: {text}")
print(f"Prediction: {result}")

