# -*- coding: utf-8 -*-
"""
Revised ABSA and sentiment-classification pipeline.

Outputs are written to ./revised_outputs/ and are intended to replace the
reviewer-questioned tables/metrics in the manuscript revision.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    STEMMER = StemmerFactory().create_stemmer()
except Exception:
    STEMMER = None

STEM_CACHE: Dict[str, str] = {}


RANDOM_STATE = 42
OUTPUT_DIR = Path("revised_outputs")
ASPECTS = ["ambiance", "packaging", "price", "service", "taste"]
BRAND_LEVEL = {
    "Starbucks": "Global Brand",
    "Kopi Kenangan": "National Brand",
    "Kopi Nako": "Local Brand",
}

MODEL_STOPWORDS = {
    "a", "ada", "adalah", "aja", "aku", "and", "atau", "buat", "dan",
    "dari", "dengan", "di", "dia", "for", "from", "i", "in", "ini", "is",
    "it", "itu", "jadi", "juga", "kami", "karena", "ke", "kita", "lagi",
    "mereka", "of", "on", "or", "saja", "saya", "sudah", "the", "this",
    "to", "untuk", "yang",
}
FREQUENCY_STOPWORDS = MODEL_STOPWORDS.union(
    {
        "area", "barista", "brand", "cafe", "coffee", "customer", "disini",
        "indonesia", "kenangan", "kopi", "kopikenangan", "kopinako", "menu",
        "nako", "ngopi", "outlet", "place", "review", "starbuck",
        "starbucks", "store", "tempat", "tempatnya",
    }
)

ASPECT_LEXICON: Dict[str, Sequence[str]] = {
    "ambiance": (
        "ambiance", "ambience", "atmosphere", "clean", "cozy", "crowded",
        "dingin", "indoor", "luas", "music", "nyaman", "outdoor",
        "parking", "parkir", "ramai", "seat", "seating", "spacious",
        "suasana", "tempat", "wifi",
    ),
    "packaging": (
        "bocor", "bungkus", "cup", "kemasan", "lid", "package",
        "packaging", "plastic", "plastik", "sedotan", "straw", "takeaway",
        "tutup",
    ),
    "price": (
        "affordable", "bayar", "biaya", "cheap", "diskon", "expensive",
        "harga", "mahal", "murah", "overpriced", "price", "pricey",
        "promo", "rp", "ribu", "terjangkau", "value", "worth",
    ),
    "service": (
        "antrian", "antre", "barista", "cashier", "cepat", "friendly",
        "hospitality", "jutek", "kasir", "lama", "layan", "layanan",
        "order", "pegawai", "pelayan", "pelayanan", "queue", "ramah",
        "rude", "service", "staff", "staf", "wait", "waiting",
    ),
    "taste": (
        "asin", "bitter", "cake", "caramel", "coffee", "delicious",
        "drink", "enak", "espresso", "food", "gula", "kopi", "latte",
        "makanan", "manis", "matcha", "minuman", "pahit", "pastry",
        "rasa", "roti", "sugar", "taste", "tawar", "yummy",
    ),
}

SENTIMENT_TERM_PATTERNS: Dict[str, Sequence[str]] = {
    "bad": ("bad", "buruk", "jelek", "tidak baik", "nggak baik", "ga baik"),
    "clean": ("clean", "bersih"),
    "cozy": ("cozy", "nyaman", "comfort", "comfortable"),
    "delicious": ("delicious", "enak", "lezat", "yummy"),
    "disappointed": ("disappointed", "disappointing", "kecewa", "mengecewakan"),
    "expensive": ("expensive", "mahal", "pricey", "overpriced"),
    "fast": ("fast", "cepat", "sigap"),
    "favorite": ("favorite", "favorit", "love", "suka"),
    "friendly": ("friendly", "ramah", "baik"),
    "good": ("good", "nice", "bagus", "mantap", "recommended", "rekomendasi"),
    "inconsistent": ("inconsistent", "beda", "tidak konsisten", "ga konsisten"),
    "less": ("kurang", "not enough"),
    "not delicious": ("tidak enak", "nggak enak", "ga enak", "gak enak", "kurang enak"),
    "not friendly": ("not friendly", "tidak ramah", "nggak ramah", "ga ramah", "gak ramah", "jutek"),
    "slow": ("slow", "lama", "lambat", "lelet", "long wait", "nunggu lama", "antri lama"),
    "spacious": ("spacious", "luas"),
    "worth": ("worth", "terjangkau", "murah", "worth it"),
}
POSITIVE_SENTIMENT_TERMS = {
    "clean",
    "cozy",
    "delicious",
    "fast",
    "favorite",
    "friendly",
    "good",
    "spacious",
    "worth",
}
NEGATIVE_SENTIMENT_TERMS = {
    "bad",
    "disappointed",
    "expensive",
    "inconsistent",
    "less",
    "not delicious",
    "not friendly",
    "slow",
}


def normalize_sentiment(value: object) -> str | None:
    if pd.isna(value):
        return None
    value_str = str(value).strip().lower()
    if value_str in {"positive", "positif", "pos"}:
        return "positive"
    if value_str in {"negative", "negatif", "neg"}:
        return "negative"
    return None


def parse_binary_flag(value: object) -> int:
    if pd.isna(value):
        return 0
    value_str = str(value).strip().lower().replace('"', "").replace(";", "")
    return 1 if value_str in {"1", "true", "yes", "y"} else 0


def clean_basic_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = re.sub(r"http\S+|www\S+", " ", str(text))
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def preprocess_for_model(text: object) -> str:
    value = clean_basic_text(text).lower()
    value = re.sub(r"[^a-zA-Z\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    if STEMMER is None:
        return value
    stemmed_tokens = []
    for token in value.split():
        if token not in STEM_CACHE:
            STEM_CACHE[token] = STEMMER.stem(token)
        stemmed_tokens.append(STEM_CACHE[token])
    return " ".join(stemmed_tokens)


def make_review_id(prefix: str, count: int) -> List[str]:
    return [f"{prefix}_{idx:04d}" for idx in range(1, count + 1)]


def load_kopi_nako() -> pd.DataFrame:
    raw = pd.read_csv("kopinako_main_analysis.csv")
    raw = raw.dropna(subset=["text", "sentiment"]).copy()
    raw["text"] = raw["text"].map(clean_basic_text)
    raw = raw[raw["text"].str.len() > 0].copy()
    raw["sentiment"] = raw["sentiment"].map(normalize_sentiment)
    raw = raw[raw["sentiment"].isin(["positive", "negative"])].copy()

    out = pd.DataFrame(
        {
            "review_id": make_review_id("nako", len(raw)),
            "brand": "Kopi Nako",
            "brand_alias": BRAND_LEVEL["Kopi Nako"],
            "stars": pd.to_numeric(raw.get("stars"), errors="coerce"),
            "text": raw["text"],
            "sentiment": raw["sentiment"],
            "source_file": "kopinako_main_analysis.csv",
        }
    )
    for aspect, col in {
        "ambiance": "ambience",
        "packaging": "packaging",
        "price": "price",
        "service": "service",
        "taste": "taste",
    }.items():
        out[f"annotated_{aspect}"] = raw[col].map(parse_binary_flag).values
    return out


def load_starbucks() -> pd.DataFrame:
    raw = pd.read_csv("starbucks_detailed_reviews.csv")
    raw = raw.dropna(subset=["text", "sentiment_category"]).copy()
    raw["sentiment"] = raw["sentiment_category"].map(normalize_sentiment)
    raw = raw[raw["sentiment"].isin(["positive", "negative"])].copy()
    raw["text"] = raw["text"].map(clean_basic_text)
    raw = raw[raw["text"].str.len() > 0].copy()

    out = pd.DataFrame(
        {
            "review_id": make_review_id("starbucks", len(raw)),
            "brand": "Starbucks",
            "brand_alias": BRAND_LEVEL["Starbucks"],
            "stars": pd.to_numeric(raw.get("stars"), errors="coerce"),
            "text": raw["text"],
            "sentiment": raw["sentiment"],
            "source_file": "starbucks_detailed_reviews.csv",
        }
    )
    for aspect, col in {
        "ambiance": "ambience",
        "packaging": "packaging;",
        "price": "price",
        "service": "service",
        "taste": "taste",
    }.items():
        out[f"annotated_{aspect}"] = raw[col].map(parse_binary_flag).values
    return out


def load_kopi_kenangan() -> pd.DataFrame:
    raw = pd.read_excel("Kopi_Kenangan.xlsx")
    raw = raw.dropna(subset=["text", "sentiment_simplified"]).copy()
    raw["sentiment"] = raw["sentiment_simplified"].map(normalize_sentiment)
    raw = raw[raw["sentiment"].isin(["positive", "negative"])].copy()
    raw["text"] = raw["text"].map(clean_basic_text)
    raw = raw[raw["text"].str.len() > 0].copy()

    out = pd.DataFrame(
        {
            "review_id": make_review_id("kenangan", len(raw)),
            "brand": "Kopi Kenangan",
            "brand_alias": BRAND_LEVEL["Kopi Kenangan"],
            "stars": pd.to_numeric(raw.get("stars"), errors="coerce"),
            "text": raw["text"],
            "sentiment": raw["sentiment"],
            "source_file": "Kopi_Kenangan.xlsx",
        }
    )
    aspect_labels = raw.get("aspect_label", pd.Series(index=raw.index, dtype="object"))
    aspect_labels = aspect_labels.astype(str).str.strip().str.lower()
    aspect_labels = aspect_labels.replace({"ambience": "ambiance"})
    for aspect in ASPECTS:
        out[f"annotated_{aspect}"] = (aspect_labels == aspect).astype(int).values
    return out


def load_clean_reviews() -> pd.DataFrame:
    reviews = pd.concat(
        [load_starbucks(), load_kopi_kenangan(), load_kopi_nako()],
        ignore_index=True,
    )
    reviews["model_text"] = reviews["text"].map(preprocess_for_model)
    reviews["is_duplicate_text_within_brand"] = reviews.duplicated(
        subset=["brand", "text"], keep=False
    )
    return reviews


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase = phrase.lower()
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def extract_aspects(text: object) -> List[str]:
    value = clean_basic_text(text).lower()
    return [
        aspect
        for aspect, keywords in ASPECT_LEXICON.items()
        if any(phrase_in_text(keyword, value) for keyword in keywords)
    ]


def build_aspect_level_dataset(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for review in reviews.itertuples(index=False):
        for aspect in extract_aspects(review.text):
            rows.append(
                {
                    "review_id": review.review_id,
                    "brand": review.brand,
                    "brand_alias": review.brand_alias,
                    "aspect": aspect,
                    "sentiment": review.sentiment,
                    "text": review.text,
                }
            )
    return pd.DataFrame(rows)


def summarize_aspect_distribution(aspect_df: pd.DataFrame) -> pd.DataFrame:
    if aspect_df.empty:
        return pd.DataFrame()

    grouped = (
        aspect_df.groupby(["brand", "brand_alias", "aspect", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    pivot = grouped.pivot_table(
        index=["brand", "brand_alias", "aspect"],
        columns="sentiment",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for sentiment in ["negative", "positive"]:
        if sentiment not in pivot.columns:
            pivot[sentiment] = 0
    pivot["aspect_total"] = pivot["negative"] + pivot["positive"]
    brand_totals = (
        aspect_df.groupby("brand").size().rename("brand_aspect_mentions").reset_index()
    )
    pivot = pivot.merge(brand_totals, on="brand", how="left")
    pivot["aspect_share_percent"] = (
        pivot["aspect_total"] / pivot["brand_aspect_mentions"] * 100
    ).round(2)
    pivot["positive_percent"] = (
        pivot["positive"] / pivot["aspect_total"].replace(0, np.nan) * 100
    ).round(2)
    pivot["negative_percent"] = (
        pivot["negative"] / pivot["aspect_total"].replace(0, np.nan) * 100
    ).round(2)
    return pivot[
        [
            "brand",
            "brand_alias",
            "aspect",
            "aspect_total",
            "brand_aspect_mentions",
            "aspect_share_percent",
            "positive",
            "negative",
            "positive_percent",
            "negative_percent",
        ]
    ].sort_values(["brand", "aspect"])


def summarize_annotated_aspects(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand, brand_df in reviews.groupby("brand"):
        brand_mentions = sum(int(brand_df[f"annotated_{aspect}"].sum()) for aspect in ASPECTS)
        for aspect in ASPECTS:
            aspect_rows = brand_df[brand_df[f"annotated_{aspect}"] == 1]
            pos = int((aspect_rows["sentiment"] == "positive").sum())
            neg = int((aspect_rows["sentiment"] == "negative").sum())
            total = pos + neg
            rows.append(
                {
                    "brand": brand,
                    "brand_alias": BRAND_LEVEL[brand],
                    "aspect": aspect,
                    "aspect_total": total,
                    "brand_aspect_mentions": brand_mentions,
                    "aspect_share_percent": round(total / brand_mentions * 100, 2)
                    if brand_mentions
                    else 0,
                    "positive": pos,
                    "negative": neg,
                    "positive_percent": round(pos / total * 100, 2) if total else 0,
                    "negative_percent": round(neg / total * 100, 2) if total else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["brand", "aspect"])


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words=sorted(MODEL_STOPWORDS),
    )


def get_models() -> Dict[str, object]:
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def get_cv_models() -> Dict[str, object]:
    """Core reviewer-requested models for stratified cross-validation."""
    models = get_models()
    return {
        "Decision Tree": models["Decision Tree"],
        "Naive Bayes": models["Naive Bayes"],
        "SVM": models["SVM"],
    }


def oversample_training_data(
    x_train: Sequence[str], y_train: Sequence[str]
) -> Tuple[pd.Series, pd.Series]:
    train_df = pd.DataFrame({"text": list(x_train), "sentiment": list(y_train)})
    counts = train_df["sentiment"].value_counts()
    if len(counts) < 2 or counts.nunique() == 1:
        return train_df["text"], train_df["sentiment"]

    majority_label = counts.idxmax()
    majority_count = int(counts.max())
    balanced_parts = [train_df[train_df["sentiment"] == majority_label]]
    for label in counts.index:
        if label == majority_label:
            continue
        minority = train_df[train_df["sentiment"] == label]
        balanced_parts.append(
            resample(
                minority,
                replace=True,
                n_samples=majority_count,
                random_state=RANDOM_STATE,
            )
        )
    balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return balanced["text"], balanced["sentiment"]


def dense_if_needed(model_name: str, matrix):
    if model_name in {"Decision Tree", "Random Forest", "Gradient Boosting"}:
        return matrix.toarray()
    return matrix


def metric_row(model_name: str, y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall_Weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1_Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision_Macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_holdout(reviews: pd.DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(
        reviews["model_text"],
        reviews["sentiment"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=reviews["sentiment"],
    )
    x_train_balanced, y_train_balanced = oversample_training_data(x_train, y_train)
    vectorizer = build_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train_balanced)
    x_test_tfidf = vectorizer.transform(x_test)

    metric_rows = []
    class_report_rows = []
    confusion_rows = []
    predictions = {}

    for model_name, model in get_models().items():
        fitted = clone(model)
        fitted.fit(dense_if_needed(model_name, x_train_tfidf), y_train_balanced)
        y_pred = fitted.predict(dense_if_needed(model_name, x_test_tfidf))
        predictions[model_name] = y_pred

        row = metric_row(model_name, y_test, y_pred)
        row["Train_Size_Original"] = len(x_train)
        row["Train_Size_After_Oversampling"] = len(x_train_balanced)
        row["Test_Size"] = len(x_test)
        metric_rows.append(row)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        for label, scores in report.items():
            if isinstance(scores, dict):
                class_report_rows.append(
                    {
                        "Model": model_name,
                        "Class": label,
                        "Precision": scores.get("precision"),
                        "Recall": scores.get("recall"),
                        "F1-Score": scores.get("f1-score"),
                        "Support": scores.get("support"),
                    }
                )

        labels = ["negative", "positive"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        for actual_idx, actual_label in enumerate(labels):
            for pred_idx, pred_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "Model": model_name,
                        "Actual": actual_label,
                        "Predicted": pred_label,
                        "Count": int(cm[actual_idx, pred_idx]),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        "F1_Weighted", ascending=False
    )
    return (
        metrics_df,
        pd.DataFrame(class_report_rows),
        pd.DataFrame(confusion_rows),
        pd.Series(y_test, name="sentiment"),
        predictions,
    )


def evaluate_cross_validation(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    x = reviews["model_text"].reset_index(drop=True)
    y = reviews["sentiment"].reset_index(drop=True)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y.iloc[test_idx]
        x_train_balanced, y_train_balanced = oversample_training_data(x_train, y_train)

        vectorizer = build_vectorizer()
        x_train_tfidf = vectorizer.fit_transform(x_train_balanced)
        x_test_tfidf = vectorizer.transform(x_test)

        for model_name, model in get_cv_models().items():
            fitted = clone(model)
            fitted.fit(dense_if_needed(model_name, x_train_tfidf), y_train_balanced)
            y_pred = fitted.predict(dense_if_needed(model_name, x_test_tfidf))
            row = metric_row(model_name, y_test, y_pred)
            row["Fold"] = fold_idx
            rows.append(row)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUTPUT_DIR / "revised_model_cv_folds.csv", index=False)
    return (
        fold_df.groupby("Model")
        .agg(
            Accuracy_Mean=("Accuracy", "mean"),
            Accuracy_Std=("Accuracy", "std"),
            F1_Weighted_Mean=("F1_Weighted", "mean"),
            F1_Weighted_Std=("F1_Weighted", "std"),
            F1_Macro_Mean=("F1_Macro", "mean"),
            F1_Macro_Std=("F1_Macro", "std"),
            Precision_Weighted_Mean=("Precision_Weighted", "mean"),
            Recall_Weighted_Mean=("Recall_Weighted", "mean"),
        )
        .reset_index()
        .sort_values("F1_Weighted_Mean", ascending=False)
    )


def sentiment_words_by_brand(reviews: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []
    for (brand, sentiment), group in reviews.groupby(["brand", "sentiment"]):
        counter: Counter[str] = Counter()
        allowed_terms = (
            POSITIVE_SENTIMENT_TERMS
            if sentiment == "positive"
            else NEGATIVE_SENTIMENT_TERMS
        )
        for text in group["text"]:
            normalized = clean_basic_text(text).lower()
            normalized = re.sub(r"[^a-zA-Z\s]", " ", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            for term, patterns in SENTIMENT_TERM_PATTERNS.items():
                if term not in allowed_terms:
                    continue
                for pattern in patterns:
                    matches = re.findall(rf"\b{re.escape(pattern)}\b", normalized)
                    counter[term] += len(matches)
        for rank, (word, count) in enumerate(counter.most_common(top_n), start=1):
            if count == 0:
                continue
            rows.append(
                {
                    "brand": brand,
                    "brand_alias": BRAND_LEVEL[brand],
                    "sentiment": sentiment,
                    "rank": rank,
                    "word": word,
                    "count": count,
                }
            )
    return pd.DataFrame(rows).sort_values(["brand", "sentiment", "rank"])


def save_plots(holdout_metrics: pd.DataFrame, confusion_df: pd.DataFrame, aspect_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=holdout_metrics, y="Model", x="F1_Weighted", ax=ax, color="#4C78A8")
    ax.set_xlim(0, 1)
    ax.set_title("Revised Holdout Weighted F1 by Model")
    ax.set_xlabel("Weighted F1")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revised_model_f1_comparison.png", dpi=200)
    plt.close(fig)

    for model_name in ["Decision Tree", holdout_metrics.iloc[0]["Model"]]:
        model_cm = confusion_df[confusion_df["Model"] == model_name]
        if model_cm.empty:
            continue
        matrix = model_cm.pivot(index="Actual", columns="Predicted", values="Count")
        matrix = matrix.reindex(index=["negative", "positive"], columns=["negative", "positive"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(OUTPUT_DIR / f"revised_confusion_matrix_{safe_name}.png", dpi=200)
        plt.close(fig)

    if not aspect_summary.empty:
        heatmap_df = aspect_summary.pivot_table(
            index="brand_alias",
            columns="aspect",
            values="positive_percent",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        ax.set_title("Positive Sentiment Percent by Aspect")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "revised_aspect_positive_heatmap.png", dpi=200)
        plt.close(fig)


def write_summary(
    reviews: pd.DataFrame,
    aspect_df: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    cv_summary: pd.DataFrame,
    frequency_table: pd.DataFrame,
) -> None:
    counts = reviews.groupby(["brand", "brand_alias", "sentiment"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    dt = holdout_metrics[holdout_metrics["Model"] == "Decision Tree"].iloc[0]
    best = holdout_metrics.iloc[0]
    cv_best = cv_summary.iloc[0]

    lines = [
        "# Revised Analysis Summary",
        "",
        "## Data",
        "",
        f"- Final standardized review rows: {len(reviews):,}",
        f"- Lexicon-generated aspect rows: {len(aspect_df):,}",
        f"- PySastrawi stemming used: {'yes' if STEMMER is not None else 'no; fallback preprocessing used'}",
        "",
        counts.reset_index().to_markdown(index=False),
        "",
        "## Holdout Model Results",
        "",
        holdout_metrics[
            ["Model", "Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "F1_Macro"]
        ].round(4).to_markdown(index=False),
        "",
        "## Five-Fold Cross-Validation",
        "",
        "CV is reported for the reviewer-requested core models: Decision Tree, Naive Bayes, and SVM.",
        "",
        cv_summary.round(4).to_markdown(index=False),
        "",
        "## Decision Tree Metrics for Revised Table 1",
        "",
        f"- Accuracy: {dt['Accuracy']:.4f}",
        f"- Weighted precision: {dt['Precision_Weighted']:.4f}",
        f"- Weighted recall: {dt['Recall_Weighted']:.4f}",
        f"- Weighted F1-score: {dt['F1_Weighted']:.4f}",
        f"- Macro F1-score: {dt['F1_Macro']:.4f}",
        "",
        "## Recommended Manuscript Notes",
        "",
        f"- Best holdout model by weighted F1: {best['Model']} ({best['F1_Weighted']:.4f}).",
        f"- Best 5-fold CV model by mean weighted F1: {cv_best['Model']} ({cv_best['F1_Weighted_Mean']:.4f}).",
        "- Describe Decision Tree as an interpretable baseline, not as state of the art.",
        "- Oversampling is applied only to training partitions to prevent duplicate-review leakage into the test set.",
        "- Use revised_sentiment_word_frequency.csv for Tables 5-7; neutral domain nouns such as coffee/kopi are filtered.",
        "",
        "## Files To Use",
        "",
        "- revised_clean_reviews.csv",
        "- revised_absa_aspect_level.csv",
        "- revised_aspect_distribution_table.csv",
        "- revised_model_holdout_metrics.csv",
        "- revised_model_cv_summary.csv",
        "- revised_classification_report.csv",
        "- revised_confusion_matrices.csv",
        "- revised_sentiment_word_frequency.csv",
        "",
        "## Top Sentiment Words Preview",
        "",
        frequency_table.head(24).to_markdown(index=False),
        "",
    ]
    (OUTPUT_DIR / "REVISED_ANALYSIS_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading and standardizing all three brand datasets...")
    reviews = load_clean_reviews()
    reviews.to_csv(OUTPUT_DIR / "revised_clean_reviews.csv", index=False)

    print("Building explicit rule-based ABSA aspect rows...")
    aspect_df = build_aspect_level_dataset(reviews)
    aspect_df.to_csv(OUTPUT_DIR / "revised_absa_aspect_level.csv", index=False)
    aspect_summary = summarize_aspect_distribution(aspect_df)
    aspect_summary.to_csv(OUTPUT_DIR / "revised_aspect_distribution_table.csv", index=False)

    annotated_summary = summarize_annotated_aspects(reviews)
    annotated_summary.to_csv(
        OUTPUT_DIR / "revised_annotated_aspect_distribution_reference.csv",
        index=False,
    )
    pd.DataFrame(
        [{"brand": brand, "brand_alias": alias} for brand, alias in BRAND_LEVEL.items()]
    ).to_csv(OUTPUT_DIR / "revised_brand_alias_mapping.csv", index=False)

    print("Evaluating holdout models with training-only oversampling...")
    holdout_metrics, class_report, confusion_df, _, _ = evaluate_holdout(reviews)
    holdout_metrics.to_csv(OUTPUT_DIR / "revised_model_holdout_metrics.csv", index=False)
    class_report.to_csv(OUTPUT_DIR / "revised_classification_report.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "revised_confusion_matrices.csv", index=False)
    holdout_metrics[holdout_metrics["Model"] == "Decision Tree"].to_csv(
        OUTPUT_DIR / "revised_table1_decision_tree_metrics.csv",
        index=False,
    )

    print("Running 5-fold stratified cross-validation...")
    cv_summary = evaluate_cross_validation(reviews)
    cv_summary.to_csv(OUTPUT_DIR / "revised_model_cv_summary.csv", index=False)

    print("Generating sentiment-bearing word frequency tables...")
    frequency_table = sentiment_words_by_brand(reviews, top_n=8)
    frequency_table.to_csv(OUTPUT_DIR / "revised_sentiment_word_frequency.csv", index=False)

    print("Saving plots and summary...")
    save_plots(holdout_metrics, confusion_df, aspect_summary)
    write_summary(reviews, aspect_df, holdout_metrics, cv_summary, frequency_table)

    print("\nRevised pipeline complete.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(
        holdout_metrics[
            ["Model", "Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
