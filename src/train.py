# src/train.py

import os
import argparse
from datasets import load_dataset
from model_setup import get_model_and_tokenizer
import tensorflow as tf
from transformers import DataCollatorForSeq2Seq

# --- Configuration ---
DATASET_NAME = "deccan-ai/insuranceQA-v2"
MODEL_CHECKPOINT = "t5-small"

def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenizes the input and output text for T5."""
    prefix = "question: "
    inputs = [prefix + doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["output"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(args):
    """Main function to run the training pipeline."""
    
    # --- 1. Load Model and Tokenizer ---
    model, tokenizer = get_model_and_tokenizer(MODEL_CHECKPOINT)
    if not model or not tokenizer:
        print("Exiting due to model loading failure.")
        return

    # --- 2. Load and Preprocess Dataset ---
    print("Loading and preprocessing dataset...")
    raw_dataset = load_dataset(DATASET_NAME)
    
    # Get the column names of the raw dataset to remove them after mapping
    original_columns = raw_dataset["train"].column_names
    
    tokenized_datasets = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        # *** THIS IS THE FIX ***
        # Remove the original text columns after tokenization
        remove_columns=original_columns
    )
    
    # --- 3. Prepare for Training ---
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
    
    train_dataset = tokenized_datasets["train"].to_tf_dataset(
        batch_size=args.batch_size,
        columns=["input_ids", "attention_mask", "labels"], # Be explicit about columns
        label_cols="labels",
        shuffle=True,
        collate_fn=data_collator,
    )

    validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
        batch_size=args.batch_size,
        columns=["input_ids", "attention_mask", "labels"], # Be explicit about columns
        label_cols="labels",
        shuffle=False,
        collate_fn=data_collator,
    )
    
    # --- 4. Compile and Train ---
    print("Compiling and training the model...")
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
    # The T5 model has a built-in loss function, so we don't need to specify one
    model.compile(optimizer=optimizer)

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs
    )

    # --- 5. Save the Model ---
    output_dir = os.path.join("models", args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a T5 model for InsuranceQA.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--model_name", type=str, required=True, help="Name to save the trained model (e.g., 'baseline_model').")
    
    args = parser.parse_args()
    main(args)