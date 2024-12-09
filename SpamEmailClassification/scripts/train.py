import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

# Evaluate model
accuracy = model.score(x_test, y_test)
# Save model and vectorizer
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)