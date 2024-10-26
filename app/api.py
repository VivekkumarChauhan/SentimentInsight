from flask import Flask, jsonify, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/sentiment_model.pt')
tokenizer = BertTokenizer.from_pretrained('models/tokenizer/')

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
