
import pandas as pd

df = pd.read_csv("arabic_txt_data.csv", encoding="utf-8")
print(df.head())
print(f'Number : {len(df)}')
print(df['label'].value_counts())
#new code
#Hi This is Haton code
import  nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
df['tokens'] = df['text'].apply(word_tokenize)
print(df['tokens'])
#Stop word remove
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_word=set(stopwords.words('arabic'))
df['filtered_tokens'] = df['tokens'].apply(
    lambda tokens: [word for word in tokens if word not in stop_word]
)
print(df['filtered_tokens'])

