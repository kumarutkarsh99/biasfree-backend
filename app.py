from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import nltk
import os
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)
CORS(app)

# -----------------------
# NLTK setup
# -----------------------

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -----------------------
# Device setup
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load model
# -----------------------

MODEL_NAME = "kumarutkarsh99/biasfree"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512
)

print("Model loaded successfully.")

# -----------------------
# Bias detection function
# -----------------------

def identify_biased_sentences(text, threshold=0.3):

    sentences = sent_tokenize(text)

    predictions = classifier(sentences)

    results = []

    for sentence, pred in zip(sentences, predictions):

        label = pred["label"]
        score = pred["score"]

        is_biased = 1 if label == "LABEL_1" and score > threshold else 0

        results.append({
            "sentence": sentence,
            "bias_score": round(score, 4),
            "label": "BIAS" if is_biased else "NEUTRAL"
        })

    return results

# -----------------------
# Health endpoint
# -----------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -----------------------
# Analyze endpoint
# -----------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "text" in request.form:

            text = request.form["text"].strip()

            if not text:
                return jsonify({"error": "Text input is empty"}), 400

            results = identify_biased_sentences(text)

            return jsonify({"results": results})

        if "file" in request.files:

            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)

            elif file.filename.endswith(".json"):
                df = pd.read_json(file)

            else:
                return jsonify({"error": "Unsupported file type"}), 400

            all_results = []

            for row in df.iloc[:, 0].dropna():

                row_results = identify_biased_sentences(str(row))

                all_results.extend(row_results)

            return jsonify({"results": all_results})

        return jsonify({"error": "No valid input provided"}), 400

    except Exception as e:

        print("Server error:", str(e))

        return jsonify({"error": str(e)}), 500

# -----------------------
# Run server
# -----------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)