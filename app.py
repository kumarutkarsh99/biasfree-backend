from flask import Flask, request, jsonify
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = "kumarutkarsh99/biasfree"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model successfully loaded from Hugging Face.")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

def identify_biased_sentences(text, classifier, threshold=0.3):
    if classifier is None:
        return []
    
    sentences = sent_tokenize(text)
    results = []
    
    for sentence in sentences:
        result = classifier(sentence)
        label = result[0]['label']
        score = result[0]['score']
        is_biased = 1 if label == 'LABEL_1' and score > threshold else 0
        results.append({
            "sentence": sentence, 
            "bias_score": round(score, 4), 
            "label": "BIAS" if is_biased else "NEUTRAL"
        })
    
    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'text' in request.form:
            text = request.form['text'].strip()
            if not text:
                return jsonify({"error": "Text input is empty"}), 400
            
            results = identify_biased_sentences(text, classifier)
            return jsonify({"results": results})
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file, encoding='utf-8')
                elif file.filename.endswith('.json'):
                    df = pd.read_json(file, encoding='utf-8')
                else:
                    return jsonify({"error": "Unsupported file type. Use CSV or JSON"}), 400
                
                if df.empty or df.shape[1] == 0:
                    return jsonify({"error": "File is empty or has no columns"}), 400
                
                all_results = []
                for row in df.iloc[:, 0]:
                    row_results = identify_biased_sentences(str(row), classifier)
                    all_results.extend(row_results)
                
                return jsonify({"results": all_results})
            except Exception as e:
                return jsonify({"error": f"File processing failed: {str(e)}"}), 400
        
        return jsonify({"error": "No valid input provided"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
