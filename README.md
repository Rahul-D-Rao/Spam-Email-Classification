# Spam Email Classification Using NLP and Machine Learning

## Overview
This project aims to classify emails as either **Spam** or **Ham** using advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system preprocesses email text, extracts meaningful features, and employs robust classifiers to detect spam efficiently. An interactive web app provides real-time email classification.

## Key Features
- **Data Preprocessing**:
  - Cleans email text by removing stopwords, punctuation, and special characters.
  - Tokenization, lemmatization, and normalization.
- **Feature Extraction**:
  - Uses TF-IDF with unigram and bigram analysis for effective feature representation.
- **Machine Learning Models**:
  - Multinomial Naive Bayes and Random Forest classifiers.
  - Automatically selects the best-performing model during training.
- **Interactive Web App**:
  - Built with Streamlit for real-time email classification.
- **Performance Metrics**:
  - Achieves over 97% accuracy with optimized models.

### To Run the main streamlit application we use the below command:
```bash
streamlit run main.py
```
