"""
Fine-tune a DistilBERT model for movie review sentiment classification.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
MODEL_OUT = os.path.join(MODELS_DIR, "transformer_model")

os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
MAX_LEN = 256


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # For speed on CPU, sample subset (remove sampling for full training on GPU)
    train_df = train_df.sample(n=min(2000, len(train_df)), random_state=42).reset_index(drop=True)
    test_df = test_df.sample(n=min(500, len(test_df)), random_state=42).reset_index(drop=True)

    y_train = (train_df["sentiment"] == "positive").astype(int).tolist()
    y_test = (test_df["sentiment"] == "positive").astype(int).tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = ReviewDataset(train_df["review"].tolist(), y_train, tokenizer)
    test_dataset = ReviewDataset(test_df["review"].tolist(), y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    # Save model
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o")
    plt.title("Training Loss Curve (DistilBERT)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_loss.png"))
    plt.close()
    print(f"Loss curve saved to: {os.path.join(PLOTS_DIR, 'training_loss.png')}")


if __name__ == "__main__":
    main()
