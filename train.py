import pandas as pd
from model import train_model, predict_text

df = pd.read_csv("arabic_txt_data.csv")

# تنظيف
df = df.dropna()
df = df.drop_duplicates()
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

print("عدد البيانات:", len(df))
print(df["label"].value_counts())

print("\nبدأ التدريب...")
train_model(df["text"], df["label"])

print("\nاختبارات:")
print(predict_text("شربت كوب ماء"))
print(predict_text("هذا الشيء بقهرني"))
print(predict_text("مش عارفة أركز"))
print(predict_text("ما في فايدة من اللي بعمله"))