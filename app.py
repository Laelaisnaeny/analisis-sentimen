from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and tokenizer for IndoBERT
tokenizer = BertTokenizer.from_pretrained('./sentiment-model')
model = BertForSequenceClassification.from_pretrained('./sentiment-model')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set Indonesian stopwords
stop_words = set(stopwords.words('indonesian'))

# Read dataset without headers from a TSV file
df = pd.read_csv('dataset.tsv', sep='\t', header=None, names=['text', 'label'])

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer.encode_plus(
        preprocessed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    outputs = model(**inputs)
    scores = outputs.logits[0].detach().numpy()
    scores = torch.softmax(torch.tensor(scores), dim=0).numpy()

    sentiment_scores = [float(score) for score in scores]
    sentiment_type = ["Negative", "Neutral", "Positive"][np.argmax(sentiment_scores)]

    return sentiment_scores, sentiment_type

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/sentimen')
def analyze():
    return render_template('input_form.html')
    
@app.route('/evaluasi')
def evaluasi():
    return render_template('evaluasi.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    scores, sentiment = predict_sentiment(text)
    return jsonify({"scores": scores, "sentiment": sentiment, "text": text, "preprocess_text": preprocess_text(text)})

@app.route('/sentiment_scores')
def sentiment_scores():
    sentiments = []
    for idx, row in df.iterrows():
        text = row['text']
        scores, sentiment = predict_sentiment(text)
        sentiments.append({"text": text, "scores": scores, "sentiment": sentiment})
    
    return jsonify({"sentiments": sentiments})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
