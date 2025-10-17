import pandas as pd
from collections import Counter
import nltk
import re
import os
from datasets import load_dataset

TOP_N_KEYWORDS = 50
DATASET_NAME = "deccan-ai/insuranceQA-v2"
OUTPUT_FILE = "data/processed/keywords.txt"

def generate_keyword_list():
    """
    Analyzes the training data from Hugging Face to extract the most 
    common keywords and saves them to a file.
    """
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    raw_dataset = load_dataset(DATASET_NAME)
    df = raw_dataset['train'].to_pandas()
    
    full_text = ' '.join(df['output'].dropna()).lower()
    
    words = re.findall(r'\b\w+\b', full_text)
    
    filtered_words = [
        word for word in words if word.isalpha() and word not in stop_words
    ]
    
    word_counts = Counter(filtered_words)
    
    top_keywords = [word for word, count in word_counts.most_common(TOP_N_KEYWORDS)]
    
    print(f"Extracted {len(top_keywords)} keywords. Saving to {OUTPUT_FILE}...")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for keyword in top_keywords:
            f.write(f"{keyword}\n")
            
    print("Keyword generation complete!")

if __name__ == '__main__':
    generate_keyword_list()