import argparse
import tensorflow as tf
from datasets import load_dataset
from model_setup import get_model_and_tokenizer
import run_evaluation
from tqdm import tqdm

def main(args):
    """
    Evaluates a fine-tuned model on a given dataset split using BLEU score.
    """
    model, tokenizer = get_model_and_tokenizer(args.model_dir)
    if not model or not tokenizer:
        return

    print(f"Loading '{args.split}' split of the dataset...")
    raw_dataset = load_dataset("deccan-ai/insuranceQA-v2", split=args.split)
    
    print("Generating predictions for the dataset...")
    predictions = []
    references = []

    for item in tqdm(raw_dataset):
        question = item['input']
        reference_answer = item['output']
        
        prefixed_input = "question: " + question
        inputs = tokenizer(prefixed_input, return_tensors="tf").input_ids
        
        outputs = model.generate(
            inputs, max_length=512, num_beams=5, early_stopping=True
        )
        
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(prediction_text)
        references.append(reference_answer)

    print("Calculating BLEU score...")
    bleu_metric = run_evaluation.load("bleu")
    results = bleu_metric.compute(predictions=predictions, references=references)

    print("\n--- Evaluation Results ---")
    print(f"Model: {args.model_dir}")
    print(f"Dataset Split: {args.split}")
    print(f"BLEU Score: {results['bleu']:.4f}")
    print("--------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a T5 model.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models/lr_experiment_model", 
        help="Directory of the fine-tuned model to evaluate."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (e.g., 'test' or 'validation')."
    )
    args = parser.parse_args()
    main(args)