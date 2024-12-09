import os
import pickle

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

def test_model_files():
    assert os.path.exists(MODEL_PATH), f"Model file missing at {MODEL_PATH}"
    assert os.path.exists(VECTORIZER_PATH), f"Vectorizer file missing at {VECTORIZER_PATH}"
    print("Model file tests passed.")

if __name__ == "__main__":
    test_model_files()