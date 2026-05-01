import streamlit as st
import base64
from model import predict_text, load_model

st.set_page_config(page_title="Arabic Emotion Detector", layout="centered")

# ================= الخلفية =================
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("bg.jpg")

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image:
        linear-gradient(rgba(0,0,0,0), rgba(0,0,0,0.4)),
        url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

.block-container {{
    padding-top: 2rem !important;
}}

h1, p, label {{
    color: white !important;
}}

textarea {{
    border-radius: 10px !important;
}}
</style>
""", unsafe_allow_html=True)

# ================= تحميل المودل =================
@st.cache_resource
def load_saved_model():
    load_model()

load_saved_model()

# ================= الواجهة =================
st.markdown("<h1 style='text-align:center;'>🧠 محلل المشاعر العربي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>اكتب جملة وسيتم تحليل الشعور</p>", unsafe_allow_html=True)

user_input = st.text_area("✍️ اكتب النص هنا:")

if st.button("🔍 تحليل"):
    if user_input.strip() == "":
        st.warning("⚠️ الرجاء إدخال نص")
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