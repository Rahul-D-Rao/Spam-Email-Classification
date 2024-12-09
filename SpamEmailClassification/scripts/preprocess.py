import os
import pandas as pd
import re

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed.csv")

def preprocess_text(text):
    """Clean and preprocess text."""
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip().lower()

def preprocess_data():
    """Load, preprocess, and save the dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    
    data = pd.read_csv(DATA_PATH)
    data["Message"] = data["Message"].apply(preprocess_text)
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved at {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()