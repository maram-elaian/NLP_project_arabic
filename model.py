from sklearn.naive_bayes import MultinomialNB

# موديل مؤقت
model = MultinomialNB()

def train_model(X_train, y_train):
    model.fit(X_train, y_train)

def predict(text_vector):
    return model.predict(text_vector)

def evaluate(X_test, y_test):
    return model.score(X_test, y_test)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["أنا متوتر", "هذا رائع", "أنا معصب"]
labels = ["توتر", "عادي", "غضب"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels)

train_model(X_train, y_train)

print(evaluate(X_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
def predict_text(text, vectorizer):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]
#Hi I'm Haton
