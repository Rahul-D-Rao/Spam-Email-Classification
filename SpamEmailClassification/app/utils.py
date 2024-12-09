import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the correct path is used for reading and writing files
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))  # Get the path of the current script

def get_model_path(model_name):
    return os.path.join(get_project_root(), '..', 'models', model_name)

def get_data_path(file_name):
    return os.path.join(get_project_root(), '..', 'data', file_name)

def clean_text(text):
    """
    Cleans the text by removing special characters, stop words, etc.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def preprocess_email_data(df):
    """
    Preprocesses the emails by cleaning and tokenizing the text.
    """
    df['Message'] = df['Message'].apply(clean_text)
    return df

def get_vectorizer(df):
    """
    Fits a TfidfVectorizer on the email dataset.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(df['Message'])
    return vectorizer

def vectorize_text(text, vectorizer):
    """
    Converts the text into a vector using the provided vectorizer.
    """
    return vectorizer.transform([text])
