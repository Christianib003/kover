# src/train.py

import os
import argparse
from datasets import load_dataset
from model_setup import get_model_and_tokenizer
import tensorflow as tf
from transformers import DataCollatorForSeq2Seq, create_optimizer

def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenizes the input and output text for T5."""
    prefix = "question: "
    inputs = [prefix + doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    
    labels = tokenizer(text_target=examples["output"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(args):
    """Main function to run the training pipeline."""
    
    # --- 1. Load Model and Tokenizer ---
    model, tokenizer = get_model_and_tokenizer("t5-small")
    if not model or not tokenizer:
        print("Exiting due to model loading failure.")
        return

    # --- 2. Load and Preprocess Dataset ---
    print("Loading and preprocessing dataset...")
    raw_dataset = load_dataset("deccan-ai/insuranceQA-v2")
    original_columns = raw_dataset["train"].column_names
    
    tokenized_datasets = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=original_columns
    )
    
    # --- 3. Prepare for Training ---
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
    
    train_dataset = tokenized_datasets["train"].to_tf_dataset(
        batch_size=args.batch_size,
        columns=["input_ids", "attention_mask", "labels"],
        label_cols="labels",
        shuffle=True,
        collate_fn=data_collator,
    )

    validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
        batch_size=args.batch_size,
        columns=["input_ids", "attention_mask", "labels"],
        label_cols="labels",
        shuffle=False,
        collate_fn=data_collator,
    )
    
    # --- 4. Compile and Train with a Scheduler ---
    print("Compiling and training the model with a learning rate scheduler...")
    
    # Calculate the total number of training steps for the scheduler
    num_train_steps = len(train_dataset) * args.epochs
    
    # Use the create_optimizer function from Hugging Face to get AdamW with a linear decay scheduler
    optimizer, schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_warmup_steps=0, # No warmup needed for this task
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01, # Standard value for AdamW
    )
    
    # The T5 model has a built-in loss function, so we only need to compile the optimizer
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
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--model_name", type=str, required=True, help="Name to save the trained model.")
    
    args = parser.parse_args()
    main(args)