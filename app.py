
import streamlit as st
import joblib

# 載入模型
model = joblib.load("model_b.pkl")

st.title("📈 年營收預測小工具")
st.write("請輸入用戶行為資料，幫你預測年營收：")

# 使用者輸入欄位
freq = st.number_input("每月消費次數", 0, 30, 10)
spent = st.number_input("平均消費金額", 0, 10000, 800)
days = st.number_input("最近一次活躍距今幾天", 0, 90, 5)

# 按鈕觸發預測
if st.button("預測"):
    result = model.predict([[freq, spent, days]])[0]
    st.success(f"📊 預估年營收：約 {int(result):,} 元")
