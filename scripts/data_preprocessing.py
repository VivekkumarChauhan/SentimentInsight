# data_preprocessing.py

import re
import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
import os

# Download required NLTK resources (only the first time it runs)
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize NLP tools
nlp = spacy.load('en_core_web_sm')
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """
    Cleans the input text by:
    - Removing URLs, mentions, special characters.
    - Converting to lowercase.
    - Lemmatizing and removing stopwords.
    """
    text = re.sub(r"http\S+|@\S+|[^A-Za-z0-9]+", " ", text)  # Remove URLs, mentions, special chars
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def preprocess_data():
    """
    Loads raw tweet data, cleans the text, calculates sentiment scores,
    and saves the processed data to a new CSV file.
    """
    # Check if raw data file exists
    raw_data_path = 'data/raw_data/tweets.csv'
    if not os.path.exists(raw_data_path):
        print(f"Error: The file {raw_data_path} does not exist.")
        return

    # Load raw data
    df = pd.read_csv(raw_data_path)
    
    # Apply text cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Calculate sentiment scores using VADER
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Save processed data
    processed_data_path = 'data/cleaned_data.csv'
    df.to_csv(processed_data_path, index=False)
    print(f"Data preprocessing complete. Cleaned data saved to {processed_data_path}")

if __name__ == "__main__":
    preprocess_data()
