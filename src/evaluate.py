"""
Generate evaluation report comparing classical and transformer models.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.append(os.path.dirname(__file__))
from preprocessor import preprocess_dataset

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_classical(test_df):
    model = joblib.load(os.path.join(MODELS_DIR, "classical_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    df = preprocess_dataset(test_df.copy())
    X_test = vectorizer.transform(df["clean_review"])
    y_test = (df["sentiment"] == "positive").astype(int).values

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        probs = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        probs = model.predict(X_test).astype(float)

    preds = model.predict(X_test)
    return y_test, preds, probs


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def plot_roc_curves(results):
    plt.figure(figsize=(7, 6))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve_comparison.png"))
    plt.close()


def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).sample(n=500, random_state=42).reset_index(drop=True)

    roc_results = {}

    # Classical model
    if os.path.exists(os.path.join(MODELS_DIR, "classical_model.pkl")):
        y_true, preds, probs = evaluate_classical(test_df)
        plot_confusion_matrix(y_true, preds, "Confusion Matrix - Classical Model",
                               "confusion_matrix_classical.png")
        roc_results["Classical Model"] = (y_true, probs)

        acc = (preds == y_true).mean()
        print(f"Classical Model Accuracy: {acc:.4f}")

        print("\nSample predictions (Classical):")
        sample_df = test_df.iloc[:10].copy()
        sample_df["predicted"] = ["positive" if p == 1 else "negative" for p in preds[:10]]
        sample_df["true_label"] = sample_df["sentiment"]
        sample_df["review_snippet"] = sample_df["review"].str[:80] + "..."
        sample_df["confidence"] = [round(p, 3) for p in probs[:10]]
        print(sample_df[["review_snippet", "true_label", "predicted", "confidence"]].to_string(index=False))
    else:
        print("Classical model not found. Run train_classical.py first.")

    if roc_results:
        plot_roc_curves(roc_results)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("\nSummary:")
    print("- Run train_classical.py and train_transformer.py to populate full comparison.")
    print("- Confusion matrices and ROC curves saved for trained models.")


if __name__ == "__main__":
    main()
