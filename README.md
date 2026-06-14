# 🎬 Reel Verdict — Movie Review Sentiment Analyzer

An end-to-end NLP project that classifies movie reviews as **positive** or **negative**, using both a classical ML pipeline (TF-IDF + scikit-learn) and a fine-tuned **DistilBERT** transformer. Includes a Flask API and a film-strip themed web UI.

---

## 📁 Project Structure

```
movie-sentiment-analyzer/
├── data/
│   ├── download_data.py      # Downloads IMDB dataset -> train.csv / test.csv
│   ├── train.csv              # (generated)
│   ├── test.csv                # (generated)
│   └── plots/                   # EDA & evaluation charts (generated)
├── models/
│   ├── classical_model.pkl       # (generated)
│   ├── tfidf_vectorizer.pkl       # (generated)
│   └── transformer_model/          # (generated - fine-tuned DistilBERT)
├── src/
│   ├── preprocessor.py        # Text cleaning, stopwords, lemmatization
│   ├── eda.py                   # Exploratory data analysis + plots
│   ├── train_classical.py        # Trains LogReg / SVC / Naive Bayes via TF-IDF
│   ├── train_transformer.py       # Fine-tunes DistilBERT
│   └── evaluate.py                 # Confusion matrices, ROC curves, reports
├── app/
│   ├── app.py                  # Flask API
│   └── templates/
│       └── index.html            # Web UI
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                          │
│                                                                │
│  download_data.py ──> train.csv / test.csv ──> eda.py        │
│                                  │                            │
│                                  v                            │
│                          preprocessor.py                       │
│                      (clean -> stopwords -> lemmatize)         │
└───────────────────────────────┬───────────────────────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              v                                          v
   ┌─────────────────────┐                  ┌─────────────────────────┐
   │  train_classical.py   │                  │  train_transformer.py     │
   │  TF-IDF + LogReg/SVC/  │                  │  Fine-tuned DistilBERT      │
   │  Naive Bayes           │                  │                              │
   │  -> classical_model.pkl│                  │  -> transformer_model/       │
   │  -> tfidf_vectorizer   │                  │                              │
   └───────────┬─────────────┘                  └────────────┬─────────────┘
               │                                              │
               └──────────────────┬───────────────────────────┘
                                   v
                         ┌──────────────────┐
                         │   evaluate.py       │
                         │  confusion matrix,  │
                         │  ROC curves, report  │
                         └──────────────────┘
                                   │
                                   v
                         ┌──────────────────┐
                         │     app/app.py      │
                         │   Flask API           │
                         │  /analyze endpoint     │
                         └────────┬───────────┘
                                  │
                                  v
                         ┌──────────────────┐
                         │  index.html (UI)    │
                         │ Reel Verdict frontend │
                         └──────────────────┘
```

---

## ⚙️ Setup Instructions

### 1. Clone / extract the project and create a virtual environment

```bash
cd movie-sentiment-analyzer
python -m venv venv

# Activate it
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python data/download_data.py
```

This downloads the IMDB dataset (50,000 reviews) and creates `data/train.csv` and `data/test.csv`.

### 4. (Optional) Run exploratory data analysis

```bash
python src/eda.py
```

Generates class distribution, review length, and word cloud plots in `data/plots/`.

### 5. Train the classical ML model

```bash
cd src
python train_classical.py
```

Trains Logistic Regression, LinearSVC, and Naive Bayes, picks the best by F1-score, and saves it to `models/`.

### 6. (Optional) Train the transformer model

```bash
python train_transformer.py
```

> ⚠️ This fine-tunes DistilBERT — recommended on a machine with a GPU. By default it samples a subset of the data for faster CPU training; adjust `train_df.sample(n=...)` in the script for full-dataset training.

### 7. (Optional) Generate evaluation reports

```bash
python evaluate.py
```

### 8. Run the web app

```bash
cd ..
python app/app.py
```

Open your browser to **http://localhost:5000**

---

## 🔌 API Usage

### `POST /analyze`

**Request:**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic, great acting and story!", "model": "classical"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9123,
  "model_used": "Logistic Regression",
  "processing_time_ms": 12.4
}
```

Use `"model": "transformer"` to use the fine-tuned DistilBERT model instead.

### `GET /health`

Returns status of loaded models:
```json
{
  "status": "ok",
  "classical_model_loaded": true,
  "transformer_model_loaded": false
}
```

---

## 🐳 Run with Docker

```bash
docker build -t reel-verdict .
docker run -p 5000:5000 reel-verdict
```

> Note: train your models locally first (steps 3–6) so `models/` is populated before building — or mount it as a volume:
> ```bash
> docker run -p 5000:5000 -v $(pwd)/models:/app/models reel-verdict
> ```

---

## 📊 Model Results

After running `train_classical.py`, results will print a comparison table like:

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.886    | 0.881     | 0.892  | 0.886    |
| LinearSVC           | 0.882    | 0.879     | 0.886  | 0.882    |
| Naive Bayes         | 0.851    | 0.847     | 0.858  | 0.852    |

*(Actual values depend on your training run and data sample.)*

---

## 🧩 Tech Stack

- **ML/NLP:** scikit-learn, NLTK, Hugging Face Transformers, PyTorch
- **Backend:** Flask + Flask-CORS
- **Frontend:** HTML/CSS/JS (no framework, film-strip themed UI)
- **Data:** IMDB Movie Reviews dataset (via `datasets` library)

---

## 📝 Notes

- Run all `src/` scripts from inside the `src/` directory (they use relative imports for `preprocessor.py`).
- The Flask app gracefully handles missing models — it will return a `503` with a helpful message if you try to use a model you haven't trained yet.
- For production, set `debug=False` in `app/app.py` (already configured via `.env.example`).
