"""
Exploratory Data Analysis for the movie review dataset.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import os
from collections import Counter
import re

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    train_path = os.path.join(DATA_DIR, "train.csv")
    df = pd.read_csv(train_path)

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())

    # 1. Class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment", palette="Set2")
    plt.title("Class Distribution: Positive vs Negative")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"))
    plt.close()

    # 2. Review length distribution
    df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8, 5))
    sns.histplot(df["review_length"], bins=50, kde=True, color="steelblue")
    plt.title("Review Length Distribution (word count)")
    plt.xlabel("Number of words")
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "review_length_distribution.png"))
    plt.close()

    # 3. Word cloud of most frequent words
    stopwords = set(STOPWORDS)
    text = " ".join(df["review"].astype(str).str.lower().tolist())
    text = re.sub(r"<.*?>", " ", text)  # strip html tags
    wc = WordCloud(width=1000, height=600, background_color="white",
                    stopwords=stopwords, max_words=200).generate(text)

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words in Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "wordcloud.png"))
    plt.close()

    # Top 20 frequent words (printed)
    words = re.findall(r"\b[a-z]{3,}\b", text)
    words = [w for w in words if w not in stopwords]
    top20 = Counter(words).most_common(20)
    print("\nTop 20 most frequent words:")
    for word, count in top20:
        print(f"  {word}: {count}")

    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
