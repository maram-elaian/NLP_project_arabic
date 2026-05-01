import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from model import train_model, predict_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# تحميل ملفات nltk
nltk.download('punkt')
nltk.download('stopwords')

# قراءة البيانات
df = pd.read_csv("arabic_txt_data.csv", encoding="utf-8")

print("عدد البيانات:", len(df))
print("\nتوزيع الفئات:")
print(df['label'].value_counts())

# تقسيم الكلمات
df['tokens'] = df['text'].apply(word_tokenize)

# حذف stopwords
stop_words = set(stopwords.words('arabic'))

df['filtered_tokens'] = df['tokens'].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

# إنشاء النص المنظف
df['clean_text'] = df['filtered_tokens'].apply(lambda words: " ".join(words))

# حفظ البيانات المنظفة
df.to_csv("clean_data.csv", index=False, encoding="utf-8")

# عرض عينات من كل فئة
for label in df['label'].unique():
    print(f"\n========== {label} ==========")
    print(df[df['label'] == label]['clean_text'].head(5))

# تدريب النموذج
print("\nبدأ التدريب...")
train_model(df['clean_text'], df['label'])

# اختبار سريع
print("\nاختبارات:")
print("أنا سعيد جدا =", predict_text("أنا سعيد جدا"))
print("أنا غاضب منك =", predict_text("أنا غاضب منك"))
print("أنا متوتر جدا =", predict_text("أنا متوتر جدا"))
print("أنا محبط =", predict_text("أنا محبط"))
print("هذا مضحك =", predict_text("هذا مضحك"))

