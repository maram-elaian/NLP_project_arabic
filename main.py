
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

print(df[['text', 'clean_text']].head())

import pandas as pd
import streamlit as st
import base64

# دالة لتحويل الصورة لـ base64
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# تحديد الصورة
img = get_base64("розовые обои 🩷💗💓.jpg")

# CSS للخلفية
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: 
        linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
        url("data:image/jpg;base64,{img}");
    background-size: cover;
}}
</style>
"""
st.markdown("""
<style>

/* مربع الكتابة */
textarea {
    background-color: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* النص داخل المربع */
textarea::placeholder {
    color: #ddd !important;
}

/* عنوان الحقل */
label {
    color: white !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)
df = pd.read_csv("arabic_txt_data.csv", encoding="utf-8")
print(df.head())
#new code
import streamlit as st

st.set_page_config(page_title="Arabic Emotion Detector", layout="centered")

# عنوان
st.markdown("<h1 style='text-align: center;'>🧠 محلل المشاعر العربي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>اكتب جملة وسيتم تحليل المشاعر</p>", unsafe_allow_html=True)

# أمثلة
st.markdown("### 💡 جرب أمثلة:")
col1, col2, col3 = st.columns(3)

example = ""

with col1:
    if st.button("😏 سخرية"):
        example = "أكيد الامتحان كان سهل جدًا 🙃"

with col2:
    if st.button("😡 غضب"):
        example = "أنا معصب من الوضع"

with col3:
    if st.button("😰 توتر"):
        example = "أنا متوتر من النتيجة"

# إدخال النص
user_input = st.text_area("✍️ اكتب النص هنا:", value=example)

# تحليل
if st.button("🔍 تحليل المشاعر"):

    if user_input.strip() == "":
        st.warning("⚠️ الرجاء إدخال نص أولاً")
    else:
        # مؤقت
        result = "سخرية 😏"

        st.markdown("## 📊 النتيجة:")

        # ألوان حسب الشعور
        if "غضب" in result:
            st.error(result)
        elif "توتر" in result:
            st.warning(result)
        elif "إحباط" in result:
            st.info(result)
        elif "سخرية" in result:
            st.success(result)
        else:
            st.write(result)

        # معلومات إضافية (شكل احترافي)
        st.markdown("### 🧾 تفاصيل:")
        st.write(f"- النص: {user_input}")
        st.write("- تم التحليل باستخدام نموذج NLP")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>مشروع NLP - تحليل المشاعر العربية</p>", unsafe_allow_html=True)