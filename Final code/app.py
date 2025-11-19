import os
import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
lstm_model = load_model(os.path.join(BASE_DIR, "lstm_model.h5"))
tokenizer = pickle.load(open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb"))
logistic_model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
all_stopwords = stop_words.union(set(punctuation))


def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in all_stopwords]
    return " ".join(tokens)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None

    if request.method == "POST":
        model_choice = request.form.get("model")
        review = request.form.get("review")

        if review:
            cleaned = preprocess_text(review)

            if model_choice == "logistic":
                vec = tfidf_vectorizer.transform([cleaned])
                pred = logistic_model.predict(vec)[0]
                sentiment = "Positive" if float(pred) >= 0.5 else "Negative"

            elif model_choice == "lstm":
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=200, padding="post")
                pred = lstm_model.predict(pad)[0][0]
                sentiment = "Positive" if pred >= 0.5 else "Negative"

    return render_template("index.html", sentiment=sentiment)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)