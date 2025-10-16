# src/generate_keywords.py

import pandas as pd
from collections import Counter
import nltk
import re
import os
from datasets import load_dataset

# --- Configuration ---
TOP_N_KEYWORDS = 50
DATASET_NAME = "deccan-ai/insuranceQA-v2"
OUTPUT_FILE = "data/processed/keywords.txt"

def generate_keyword_list():
    """
    Analyzes the training data from Hugging Face to extract the most 
    common keywords and saves them to a file.
    """
    # Download stopwords from NLTK if not present
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    raw_dataset = load_dataset(DATASET_NAME)
    df = raw_dataset['train'].to_pandas()
    
    # Combine all answers into a single block of text using the correct column name
    full_text = ' '.join(df['output'].dropna()).lower()
    
    # Find all words, ignoring punctuation
    words = re.findall(r'\b\w+\b', full_text)
    
    # Filter out stopwords and non-alphabetic words
    filtered_words = [
        word for word in words if word.isalpha() and word not in stop_words
    ]
    
    # Count the frequency of the remaining words
    word_counts = Counter(filtered_words)
    
    # Get the top N most common words
    top_keywords = [word for word, count in word_counts.most_common(TOP_N_KEYWORDS)]
    
    print(f"Extracted {len(top_keywords)} keywords. Saving to {OUTPUT_FILE}...")
    
    # Save the keywords to a text file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for keyword in top_keywords:
            f.write(f"{keyword}\n")
            
    print("Keyword generation complete!")

if __name__ == '__main__':
    generate_keyword_list()