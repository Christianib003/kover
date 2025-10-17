import argparse
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

def load_keywords(filepath: str) -> set:
    """Loads a list of keywords from a text file into a set for fast lookup."""
    try:
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        print(f"Warning: Keyword file not found at {filepath}. OOD filter will be disabled.")
        return set()

class Chatbot:
    def __init__(self, model_dir: str, keywords_path: str):
        print(f"Loading model from: {model_dir}...")
        self.keywords = load_keywords(keywords_path)
        try:
            self.model = TFT5ForConditionalGeneration.from_pretrained(model_dir)
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            print("\n✅ Chatbot is ready!")
        except OSError:
            print(f"Error: Model not found at {model_dir}.")
            self.model = None

    def _is_in_domain(self, question: str) -> bool:
        """Checks if any keyword is present in the question."""
        if not self.keywords:
            return True
        return any(keyword in question.lower() for keyword in self.keywords)

    def get_response(self, user_input: str) -> str:
        """Generates a response, with a data-driven check for OOD questions."""
        if not self._is_in_domain(user_input):
            return "I apologize, but I am a specialized chatbot and can only answer questions related to insurance."
        
        prefixed_input = "question: " + user_input
        inputs = self.tokenizer(prefixed_input, return_tensors="tf").input_ids
        
        outputs = self.model.generate(
            inputs, max_length=512, num_beams=5, early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main(args):
    """Main function to start the chat loop."""
    chatbot = Chatbot(args.model_dir, args.keywords_file)
    if not chatbot.model:
        return

    print("---")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
        print("---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned T5 model.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="Directory where the fine-tuned model is saved."
    )
    parser.add_argument(
        "--keywords_file",
        type=str,
        default="data/processed/keywords.txt",
        help="Path to the keyword file for OOD detection."
    )
    args = parser.parse_args()
    main(args)