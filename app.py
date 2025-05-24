
import streamlit as st
import joblib

# 載入模型
model = joblib.load("model_b.pkl")

# 欄位名稱與權重（模型參數）
features = ["monthly_freq", "avg_spent", "last_active_days"]
weights = model.coef_
intercept = model.intercept_

st.title("📈 年營收預測小工具 + 運算解釋")
st.write("請輸入用戶行為資料，我們會告訴你模型怎麼預測出結果：")

# 使用者輸入欄位
freq = st.number_input("每月消費次數", 0, 30, 10)
spent = st.number_input("平均消費金額", 0, 10000, 800)
days = st.number_input("最近一次活躍距今幾天", 0, 90, 5)

input_values = [freq, spent, days]

# 按鈕觸發預測與解釋
if st.button("預測"):
    result = model.predict([input_values])[0]

    st.success(f"📊 預估年營收：約 {int(result):,} 元")

    # 顯示公式與細節
    st.markdown("### 🧠 模型運算過程")
    st.write("以下是模型使用的線性組合（每個欄位乘上權重）：")

    parts = []
    weighted_sum = 0

    for f, w, x in zip(features, weights, input_values):
        contrib = w * x
        weighted_sum += contrib
        st.write(f"- `{f}` × {w:.2f} × {x} = **{contrib:,.2f}**")
        parts.append(f"({w:.2f} × {x})")

    st.write(f"**總和 + 截距：{' + '.join(parts)} + {intercept:.2f} = {result:,.2f}**")
    st.info(f"模型最終預測 = 所有特徵加權總和 + 截距值")
