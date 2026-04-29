import streamlit as st
import pandas as pd
import base64
from model import train_model, predict_text


st.set_page_config(page_title="Arabic Emotion Detector", layout="centered")

# =========================
# تحميل الصورة
# =========================
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("bg.jpg")

# =========================
# CSS كامل (خلفية + Full Screen)
# =========================
st.markdown(f"""
<style>

/* الخلفية */
[data-testid="stAppViewContainer"] {{
    background-image: 
        linear-gradient(rgba(0,0,0,0), rgba(0,0,0,0.3)),
        url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    min-height: 100vh;
}}

/* إزالة الفراغات */
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}}

/* مربع الكتابة */
textarea {{
    background-color: rgba(255, 255, 255) !important;
    color: black !important;
    border-radius: 10px !important;
}}

/* النص */
h1, p, label {{
    color: black !important;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# تحميل البيانات + تدريب (مرة واحدة)
# =========================
@st.cache_resource
def load_model():
    df = pd.read_csv("clean_data.csv", encoding="utf-8")
    train_model(df['clean_text'], df['label'])

load_model()

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align:center;'>🧠 محلل المشاعر العربي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>اكتب جملة وسيتم تحليل المشاعر</p>", unsafe_allow_html=True)

user_input = st.text_area("✍️ اكتب النص هنا:")

# =========================
# Prediction
# =========================
if st.button("🔍 تحليل المشاعر"):
    if user_input.strip() == "":
        st.warning("⚠️ الرجاء إدخال نص أولاً")
    else:
        result = predict_text(user_input)

        st.markdown("## 📊 النتيجة:")

        if "غضب" in result:
            st.error(result + " 😡")
        elif "توتر" in result:
            st.warning(result + " 😰")
        elif "إحباط" in result:
            st.info(result + " 😞")
        elif "سخرية" in result:
            st.success(result + " 😏")
        else:
            st.write(result + " 🙂")