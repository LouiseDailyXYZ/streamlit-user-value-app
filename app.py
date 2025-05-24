
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

st.title("📈 年營收預測工具（支援自訂資料）")
st.write("你可以使用預設模型，或上傳自己的用戶行為資料來重新訓練模型進行預測。")

features = ["monthly_freq", "avg_spent", "last_active_days"]
model = None
intercept = 0
weights = []

use_default = st.checkbox("使用預設模型（修正版，不會預測負值）", value=True)

if use_default:
    model = joblib.load("model_b_fixed.pkl")
    weights = model.coef_
    intercept = model.intercept_
    st.info("目前使用修正後的預訓練模型")
else:
    uploaded_file = st.file_uploader("📁 上傳 CSV（需包含 monthly_freq, avg_spent, last_active_days）", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.success("✅ 資料上傳成功")
            st.dataframe(df.head())

            y = np.maximum(
                df["monthly_freq"] * 600 +
                df["avg_spent"] * 10 -
                df["last_active_days"] * 50 +
                np.random.normal(0, 3000, size=len(df)),
                0
            )
            X = df[features]

            model = LinearRegression().fit(X, y)
            weights = model.coef_
            intercept = model.intercept_
            st.success("✅ 模型訓練完成")
        except Exception as e:
            st.error(f"❌ 錯誤：{e}")

if model:
    st.markdown("### ✍️ 輸入新用戶資料進行預測：")
    freq = st.number_input("每月消費次數", 0, 30, 10)
    spent = st.number_input("平均消費金額", 0, 10000, 800)
    days = st.number_input("最近一次活躍距今幾天", 0, 90, 5)

    input_values = [freq, spent, days]

    if st.button("🚀 預測年營收"):
        raw_result = model.predict([input_values])[0]
        result = max(raw_result, 0)  # 防止負值

        st.success(f"📊 預估年營收：約 {int(result):,} 元")

        st.markdown("### 🧠 模型運算過程")
        st.write("每個欄位的貢獻值如下：")

        parts = []
        for f, w, x in zip(features, weights, input_values):
            contrib = w * x
            st.write(f"- `{f}` × {w:.2f} × {x} = **{contrib:,.2f}**")
            parts.append(f"({w:.2f} × {x})")

        st.write(f"**總和 + 截距：{' + '.join(parts)} + {intercept:.2f} = {raw_result:,.2f}**")

        st.markdown("""----
#### ℹ️ 為什麼不會出現負值？
模型雖然是線性預測，但我們將預測結果設下限為 0，以符合實際商業邏輯（年營收不會是負數）。

如果輸入資料顯示用戶長期未活躍、消費頻率與金額極低，系統會判斷其為「極低價值潛在用戶」，預估值趨近 0。
""")
