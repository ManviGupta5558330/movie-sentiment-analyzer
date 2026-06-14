"""
Flask web application for movie review sentiment analysis.
Serves predictions from both a classical ML model (TF-IDF + sklearn)
and a fine-tuned DistilBERT transformer model.
"""
import os
import sys
import time
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessor import TextPreprocessor

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
CORS(app)

preprocessor = TextPreprocessor()

# ---- Load classical model (if available) ----
classical_model = None
vectorizer = None
classical_model_name = "Classical Model"

classical_path = os.path.join(MODELS_DIR, "classical_model.pkl")
vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
name_path = os.path.join(MODELS_DIR, "classical_model_name.pkl")

if os.path.exists(classical_path) and os.path.exists(vectorizer_path):
    classical_model = joblib.load(classical_path)
    vectorizer = joblib.load(vectorizer_path)
    if os.path.exists(name_path):
        classical_model_name = joblib.load(name_path)
    print(f"Loaded classical model: {classical_model_name}")
else:
    print("Classical model not found. Run src/train_classical.py first.")

# ---- Load transformer model (if available) ----
transformer_model = None
transformer_tokenizer = None
transformer_path = os.path.join(MODELS_DIR, "transformer_model")

if os.path.exists(transformer_path) and os.listdir(transformer_path):
    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        transformer_tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_path)
        transformer_model = DistilBertForSequenceClassification.from_pretrained(transformer_path)
        transformer_model.eval()
        print("Loaded transformer model.")
    except Exception as e:
        print(f"Could not load transformer model: {e}")
else:
    print("Transformer model not found. Run src/train_transformer.py first.")


def predict_classical(review_text):
    if classical_model is None or vectorizer is None:
        raise RuntimeError("Classical model not available. Train it first with src/train_classical.py")

    clean_text = preprocessor.preprocess_pipeline(review_text)
    X = vectorizer.transform([clean_text])

    pred = classical_model.predict(X)[0]

    if hasattr(classical_model, "predict_proba"):
        confidence = float(classical_model.predict_proba(X)[0][pred])
    elif hasattr(classical_model, "decision_function"):
        score = classical_model.decision_function(X)[0]
        confidence = float(1 / (1 + abs(score)) * -1 + 1)  # rough normalization
        confidence = min(max(confidence, 0.5), 0.99)
    else:
        confidence = 0.85

    sentiment = "positive" if pred == 1 else "negative"
    return sentiment, confidence, classical_model_name


def predict_transformer(review_text):
    if transformer_model is None or transformer_tokenizer is None:
        raise RuntimeError("Transformer model not available. Train it first with src/train_transformer.py")

    import torch

    inputs = transformer_tokenizer(
        review_text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred = int(torch.argmax(probs).item())
    confidence = float(probs[pred].item())
    sentiment = "positive" if pred == 1 else "negative"
    return sentiment, confidence, "DistilBERT (fine-tuned)"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or "review" not in data:
        return jsonify({"error": "Missing 'review' field in request body."}), 400

    review_text = data["review"].strip()
    model_choice = data.get("model", "classical").lower()

    if not review_text:
        return jsonify({"error": "Review text cannot be empty."}), 400

    start_time = time.time()

    try:
        if model_choice == "transformer":
            sentiment, confidence, model_used = predict_transformer(review_text)
        else:
            sentiment, confidence, model_used = predict_classical(review_text)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "model_used": model_used,
        "processing_time_ms": elapsed_ms,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "classical_model_loaded": classical_model is not None,
        "transformer_model_loaded": transformer_model is not None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
