from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import spacy

# Load mBERT (Multilingual BERT) model and tokenizer for multilingual sentiment analysis
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

# Initialize SpaCy language detectors for text preprocessing
spacy_models = {
    "en": spacy.load("en_core_web_sm"),
    "fr": spacy.load("fr_core_news_sm"),
    "es": spacy.load("es_core_news_sm"),
    # Add more languages as needed
}

def detect_language(text):
    """Detect the language of the text and return the SpaCy model for that language."""
    for lang, nlp in spacy_models.items():
        if nlp(text):  # Assuming language models have been downloaded for the specific languages
            return lang, nlp
    return "en", spacy_models["en"]  # Default to English if detection fails

def preprocess_text(text, lang):
    """Preprocess text according to the detected language model."""
    doc = spacy_models[lang](text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def predict_sentiment(text):
    """Predict sentiment for multilingual text."""
    lang, nlp = detect_language(text)  # Detect language of the text
    cleaned_text = preprocess_text(text, lang)  # Clean and preprocess the text

    # Tokenize and predict sentiment
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    sentiment_score = torch.argmax(outputs.logits, dim=1).item()

    # Map prediction to sentiment label
    sentiment_label = {2: "positive", 1: "neutral", 0: "negative"}
    return sentiment_label[sentiment_score]

# Example function to load and analyze multilingual data
def analyze_multilingual_data(file_path="data/cleaned_data.csv"):
    """Analyze sentiment for each text entry in a multilingual dataset."""
    df = pd.read_csv(file_path)
    df['predicted_sentiment'] = df['text'].apply(predict_sentiment)
    df.to_csv("data/multilingual_sentiment_results.csv", index=False)
    print("Sentiment analysis complete. Results saved to 'data/multilingual_sentiment_results.csv'.")

if __name__ == "__main__":
    analyze_multilingual_data()
