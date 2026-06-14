"""
Download the IMDB Movie Reviews dataset and split into train/test CSVs.
"""
import pandas as pd
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    print(f"Downloading IMDB dataset from {url}...")
    df = pd.read_csv(url)

    # The dataset contains 50,000 reviews. Split 50/50 into train/test.
    train_df = df.iloc[:25000].copy()
    test_df = df.iloc[25000:].copy()

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_path = os.path.join(OUT_DIR, "train.csv")
    test_path = os.path.join(OUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain set: {train_df.shape}")
    print(train_df.head())
    print(f"\nTest set: {test_df.shape}")
    print(test_df.head())
    print(f"\nSaved to:\n  {train_path}\n  {test_path}")


if __name__ == "__main__":
    main()
