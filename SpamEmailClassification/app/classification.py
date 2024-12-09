import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Determine root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
model_path = os.path.join(root_dir, 'models', 'saved_model.pkl')
vectorizer_path = os.path.join(root_dir, 'models', 'vectorizer.pkl')

# Load model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def classify_email(message: str) -> str:
    data = [message]
    vectorized_message = vectorizer.transform(data)
    prediction = model.predict(vectorized_message)
    return "spam" if prediction[0] == 1 else "ham"