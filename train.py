import pandas as pd

df = pd.read_csv("arabic_txt_data.csv", encoding="utf-8")
print(df.head())
print(f'Number : {len(df)}')
print(df['label'].value_counts())
#new code
#Hi This is Haton code
import  nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
df['tokens'] = df['text'].apply(word_tokenize)
print(df['tokens'])
#Stop word remove
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_word=set(stopwords.words('arabic'))
df['filtered_tokens'] = df['tokens'].apply(
    lambda tokens: [word for word in tokens if word not in stop_word]
)

df['clean_text'] = df['filtered_tokens'].apply(lambda words: " ".join(words))
df.to_csv("clean_data.csv", index=False, encoding="utf-8")
print(df[['text', 'clean_text']].head())



import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from model import train_model, predict_text

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("arabic_txt_data.csv", encoding="utf-8")

# Tokenization
df['tokens'] = df['text'].apply(word_tokenize)

# Stopwords
stop_words = set(stopwords.words('arabic'))

df['filtered_tokens'] = df['tokens'].apply(
    lambda tokens: [w for w in tokens if w not in stop_words]
)


df['clean_text'] = df['filtered_tokens'].apply(lambda words: " ".join(words))


train_model(df['clean_text'], df['label'])

# تجربة
print(predict_text("أنا متوتر جدًا 😰"))
df.to_csv("clean_data.csv", index=False, encoding="utf-8")

print(df['label'].value_counts())