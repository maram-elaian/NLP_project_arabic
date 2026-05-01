import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

stop_words = set(stopwords.words('arabic'))

# تنظيف النص
def clean_text(text):
    text = str(text)

    # توحيد الأحرف
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)

    # حذف الرموز
    text = re.sub(r'[^\w\s]', '', text)

    # حذف الأرقام
    text = re.sub(r'\d+', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# الأدوات
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# تدريب
def train_model(texts, labels):
    cleaned_texts = texts.apply(clean_text)

    X = vectorizer.fit_transform(cleaned_texts)
    model.fit(X, labels)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# تحميل
def load_model():
    global model, vectorizer
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# توقع
def predict_text(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    return model.predict(X)[0]