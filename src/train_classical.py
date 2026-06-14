"""
Train classical ML models (Logistic Regression, LinearSVC, Naive Bayes)
for movie review sentiment classification using TF-IDF features.
"""
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessor import TextPreprocessor, preprocess_dataset

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = preprocess_dataset(df)
    y = (df["sentiment"] == "positive").astype(int)
    return df["clean_review"], y


def main():
    print("Loading and preprocessing data...")
    X_train_text, y_train = load_and_preprocess(os.path.join(DATA_DIR, "train.csv"))
    X_test_text, y_test = load_and_preprocess(os.path.join(DATA_DIR, "test.csv"))

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "classical_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "classical_model_name.pkl"))

    print(f"\nSaved best model and vectorizer to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
