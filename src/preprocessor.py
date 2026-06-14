"""
Text preprocessing utilities for movie review sentiment analysis.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only first run)
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Remove HTML tags, special characters, and extra whitespace."""
        text = str(text)
        text = re.sub(r"<.*?>", " ", text)            # HTML tags
        text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
        text = re.sub(r"[^a-zA-Z\s]", " ", text)      # special chars / numbers
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()      # extra whitespace
        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove English stopwords."""
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words]
        return " ".join(filtered)

    def lemmatize(self, text: str) -> str:
        """Apply WordNet lemmatization."""
        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(w) for w in words]
        return " ".join(lemmatized)

    def preprocess_pipeline(self, text: str) -> str:
        """Run full preprocessing pipeline: clean -> remove stopwords -> lemmatize."""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text


def preprocess_dataset(df, text_column="review", output_column="clean_review"):
    """Apply the preprocessing pipeline to a DataFrame column."""
    preprocessor = TextPreprocessor()
    df[output_column] = df[text_column].apply(preprocessor.preprocess_pipeline)
    return df


if __name__ == "__main__":
    import pandas as pd
    import os

    train_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    df = pd.read_csv(train_path).head(5)

    preprocessor = TextPreprocessor()
    print("Sample before/after preprocessing:\n")
    for i, row in df.iterrows():
        before = row["review"]
        after = preprocessor.preprocess_pipeline(before)
        print(f"--- Review {i+1} ---")
        print("BEFORE:", before[:200], "...")
        print("AFTER :", after[:200], "...")
        print()
