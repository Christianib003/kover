import os
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_model_and_tokenizer(model_name: str):
    """
    Loads a pre-trained T5 model and its corresponding tokenizer from Hugging Face.

    This function encapsulates the model loading process, making it easy to
    call from other parts of the application (e.g., training, inference).

    Args:
        model_name (str): The name of the pre-trained T5 model to load,
                          e.g., 't5-small'.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
               (TFT5ForConditionalGeneration, T5Tokenizer)
    """
    print(f"Loading pre-trained model: {model_name}...")
    try:
        model = TFT5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except OSError as e:
        print(f"Error: Model '{model_name}' not found. Please check the model name.")
        print(f"Original error: {e}")
        return None, None

if __name__ == '__main__':
    MODEL_CHECKPOINT = "t5-small"
    test_model, test_tokenizer = get_model_and_tokenizer(MODEL_CHECKPOINT)

    if test_model and test_tokenizer:
        print("\n--- Verification ---")
        print(f"Model Class: {test_model.__class__.__name__}")
        print(f"Tokenizer Class: {test_tokenizer.__class__.__name__}")
        print("Setup script is working correctly. You can now import this function in your training script.")