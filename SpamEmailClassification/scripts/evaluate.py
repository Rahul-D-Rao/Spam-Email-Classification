import os
import sys
import joblib

# Ensure we navigate to the root directory and add the 'app' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')  # Going up one level to the root directory
app_dir = os.path.join(root_dir, 'app')
sys.path.insert(0, app_dir)  # Add the 'app' directory to the Python path for imports
from utils import get_model_path

# Define the evaluation function
def evaluate_model():
    # Load the model and vectorizer
    model_path = get_model_path('saved_model.pkl')
    vectorizer_path = get_model_path('vectorizer.pkl')
    
    # Load model and vectorizer using joblib
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Assuming you have your test data (X_test and y_test)
    from utils import preprocess_email_data
    import pandas as pd

    # Load and preprocess the test data
    df = pd.read_csv(os.path.join(root_dir, 'data', 'spam.csv'))
    df = preprocess_email_data(df)

    # Split into features and labels for evaluation
    X_test = vectorizer.transform(df['Message'])
    y_test = df['Category'].map({'ham': 0, 'spam': 1})

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")