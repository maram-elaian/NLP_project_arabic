import streamlit as st
import pandas as pd
from model import train_model, predict_text, load_model

df = pd.read_csv("clean_data.csv", encoding="utf-8")


train_model(df['clean_text'], df['label'])
load_model()



st.set_page_config(page_title="Arabic Emotion Detector", layout="centered")

st.title("🧠 محلل المشاعر العربي")

user_input = st.text_area("✍️ اكتب النص هنا:")

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