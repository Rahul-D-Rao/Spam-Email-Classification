import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Determine root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
data_path = os.path.join(root_dir, 'data', 'spam.csv')
model_path = os.path.join(root_dir, 'models', 'saved_model.pkl')
vectorizer_path = os.path.join(root_dir, 'models', 'vectorizer.pkl')

# Load data
data = pd.read_csv(data_path, encoding="latin-1")

# Clean data
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Prepare data
X = data['message']
y = data['label']
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit models
nb_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)

# Predict using both models
nb_predictions = nb_model.predict(x_test)
rf_predictions = rf_model.predict(x_test)

# Calculate performance metrics for Naive Bayes
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

# Calculate performance metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

# Print desired output
print("Model Performance:")
print(f"\nMultinomial Naive Bayes Performance:")
print(f"Accuracy: {nb_accuracy * 100:.2f}%")
print(f"Precision: {nb_precision * 100:.2f}%")
print(f"Recall: {nb_recall * 100:.2f}%")
print(f"F1-Score: {nb_f1 * 100:.2f}%")

print(f"\nRandom Forest Performance:")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Precision: {rf_precision * 100:.2f}%")
print(f"Recall: {rf_recall * 100:.2f}%")
print(f"F1-Score: {rf_f1 * 100:.2f}%")

# Save the best-performing model (based on accuracy)
best_model = nb_model if nb_accuracy > rf_accuracy else rf_model

# Save the model and vectorizer for future use
with open(model_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("\nModel and vectorizer saved successfully.")
