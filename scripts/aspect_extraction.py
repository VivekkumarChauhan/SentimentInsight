import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm')

def extract_aspects(text):
    doc = nlp(text)
    return [(chunk.text, chunk.root.head.text) for chunk in doc.noun_chunks]

def add_aspects():
    df = pd.read_csv('data/cleaned_data.csv')
    df['aspects'] = df['cleaned_text'].apply(extract_aspects)
    df.to_csv('data/cleaned_data.csv', index=False)

if __name__ == "__main__":
    add_aspects()
