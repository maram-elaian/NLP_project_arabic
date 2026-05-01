import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(max_features=5000)
model = LogisticRegression(max_iter=1000)

def train_model(texts, labels):

    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")


def predict_text(text):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    X = vectorizer.transform([text])
    return model.predict(X)[0]


def load_model():
    global model, vectorizer
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")