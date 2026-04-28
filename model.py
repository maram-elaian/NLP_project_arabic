import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

stop_words = set(stopwords.words('arabic'))

# 🔥 نفس التنظيف لكل الحالات
def clean_text(text):
    text = str(text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ------------------- أدوات -------------------

vectorizer = TfidfVectorizer(ngram_range=(1,2))
model = LogisticRegression(max_iter=1000)

# ------------------- تدريب -------------------

def train_model(texts, labels):
    cleaned_texts = texts.apply(clean_text)

    X = vectorizer.fit_transform(cleaned_texts)
    model.fit(X, labels)

    # حفظ الموديل
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# ------------------- تحميل -------------------

def load_model():
    global model, vectorizer
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------- توقع -------------------

def predict_text(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    return model.predict(X)[0]

from model import train_model, predict_text, load_model
import pandas as pd

df = pd.read_csv("clean_data.csv", encoding="utf-8")

train_model(df['clean_text'], df['label'])
load_model()

print("🔥 STREAMLIT MODEL LOADED")

print("TEST1:", predict_text("أنا سعيد"))
print("TEST2:", predict_text("أنا غاضب"))
print("TEST3:", predict_text("أنا متوتر"))