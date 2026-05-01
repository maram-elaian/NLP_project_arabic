import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download("stopwords")

stop_words = set(stopwords.words("arabic"))
stemmer = SnowballStemmer("arabic")

vectorizer = TfidfVectorizer(max_features=5000)
model = LogisticRegression(max_iter=1000)


# ================= Preprocessing =================
def preprocess(text):
    text = text.lower()
    tokens = text.split()

    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# ================= Training =================
def train_model(texts, labels):

    texts = texts.apply(preprocess)

    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ===== Evaluation =====
    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\n📉 Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")


# ================= Prediction =================
def predict_text(text):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    text = preprocess(pd.Series([text])[0])

    X = vectorizer.transform([text])
    return model.predict(X)[0]