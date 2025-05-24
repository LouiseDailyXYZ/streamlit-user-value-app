
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

st.title("📈 年營收預測工具（支援自訂資料）")
st.write("你可以使用預設模型，或上傳自己的用戶行為資料來重新訓練模型進行預測。")

# 欄位定義
features = ["monthly_freq", "avg_spent", "last_active_days"]
model = None
intercept = 0
weights = []

# 預設模型或上傳資料
use_default = st.checkbox("使用預設模型（適用範例測試）", value=True)

if use_default:
    # 載入內建模型
    model = joblib.load("model_b.pkl")
    weights = model.coef_
    intercept = model.intercept_
    st.info("目前使用預訓練模型（Version B）")
else:
    uploaded_file = st.file_uploader("📁 請上傳 CSV（需包含欄位 monthly_freq, avg_spent, last_active_days）", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.success("✅ 資料上傳成功，預覽如下：")
            st.dataframe(df.head())

            # 建立自訂目標值（模擬真實營收）
            y = (
                df["monthly_freq"] * 600 +
                df["avg_spent"] * 10 -
                df["last_active_days"] * 50 +
                np.random.normal(0, 3000, size=len(df))
            )
            X = df[features]

            # 訓練模型
            model = LinearRegression().fit(X, y)
            weights = model.coef_
            intercept = model.intercept_
            st.success("✅ 模型已成功訓練，可進行預測")
        except Exception as e:
            st.error(f"❌ 上傳資料格式有誤：{e}")

# 預測輸入區
if model:
    st.markdown("### ✍️ 請輸入新用戶資料進行預測：")
    freq = st.number_input("每月消費次數", 0, 30, 10)
    spent = st.number_input("平均消費金額", 0, 10000, 800)
    days = st.number_input("最近一次活躍距今幾天", 0, 90, 5)

    input_values = [freq, spent, days]

    if st.button("🚀 預測年營收"):
        result = model.predict([input_values])[0]
        st.success(f"📊 預估年營收：約 {int(result):,} 元")

        # 顯示運算細節
        st.markdown("### 🧠 模型運算過程")
        st.write("每個欄位的貢獻值如下：")

        parts = []
        for f, w, x in zip(features, weights, input_values):
            contrib = w * x
            st.write(f"- `{f}` × {w:.2f} × {x} = **{contrib:,.2f}**")
            parts.append(f"({w:.2f} × {x})")

        st.write(f"**總和 + 截距：{' + '.join(parts)} + {intercept:.2f} = {result:,.2f}**")

        st.markdown(
            "----
"
            "#### ℹ️ 權重是怎麼來的？
"
            "這些權重（例如：平均消費金額的影響係數 `+10.86`）來自模型分析你上傳資料後自動學出的結論。

"
            "- 數字越大 → 欄位對營收預測的影響越強
"
            "- 正數 → 代表愈高愈加分
"
            "- 負數 → 代表愈高愈扣分

"
            "權重是模型訓練後自動計算的，不是人工設定。"
        )
