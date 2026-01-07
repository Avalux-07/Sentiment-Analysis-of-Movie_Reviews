# data_loading_preprocessing & vectorization

import os
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


def load_reviews_from_folder(folder_path, label):
    texts = []
    labels = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
            labels.append(label)

    return texts, labels


def load_and_vectorize_data():
    dataset_dir = BASE_DIR / "clean_dataset"

    # ---- TRAIN DATA ----
    train_pos = dataset_dir / "train" / "pos"
    train_neg = dataset_dir / "train" / "neg"

    X_pos, y_pos = load_reviews_from_folder(train_pos, 1)
    X_neg, y_neg = load_reviews_from_folder(train_neg, 0)

    X_train = X_pos + X_neg
    y_train = y_pos + y_neg

    # ---- TEST DATA ----
    test_pos = dataset_dir / "test" / "pos"
    test_neg = dataset_dir / "test" / "neg"

    X_pos, y_pos = load_reviews_from_folder(test_pos, 1)
    X_neg, y_neg = load_reviews_from_folder(test_neg, 0)

    X_test = X_pos + X_neg
    y_test = y_pos + y_neg

    # ---- TF-IDF ----
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=5
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf

print("Data loading and preprocessing complete.")
