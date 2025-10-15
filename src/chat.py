# src/chat.py

import argparse
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

def main(args):
    """Loads a fine-tuned model and allows user interaction."""
    
    # --- 1. Load the fine-tuned model and tokenizer ---
    print(f"Loading model from: {args.model_dir}")
    try:
        model = TFT5ForConditionalGeneration.from_pretrained(args.model_dir)
        tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    except OSError:
        print(f"Error: Model not found at {args.model_dir}. Make sure the directory is correct.")
        return

    print("\n✅ Chatbot is ready! Type 'exit' to end the conversation.")
    print("---")
    
    # --- 2. Start the conversation loop ---
    while True:
        # Get user input from the terminal
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
            
        # --- 3. Prepare the input for the model ---
        # We use the same prefix as in training
        prefixed_input = "question: " + user_input
        inputs = tokenizer(prefixed_input, return_tensors="tf").input_ids
        
        # --- 4. Generate a response ---
        # Using parameters for better quality generation
        outputs = model.generate(
            inputs, 
            max_length=512, 
            num_beams=5, # beam search helps find better answers
            early_stopping=True
        )
        
        # --- 5. Decode and print the response ---
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Chatbot: {response}")
        print("---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned T5 model.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models/baseline_model", 
        help="Directory where the fine-tuned model is saved."
    )
    args = parser.parse_args()
    main(args)