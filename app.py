import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 모델 및 스케일러 로딩
model = load_model("mlp_model_3to3.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 앱 제목
st.title("💠 에어로겔 물성 예측기")

# 입력 폼
with st.form("input_form"):
    frequency = st.text_input("🔹 Frequency (Hz) (10 ~ 100000)", value="1000")
    impedance = st.text_input("🔹 Impedance (Ω) (1000 ~ 50000)", value="5000")
    time = st.text_input("🔹 Time (분) (0 ~ 1440)", value="60")
    
    submitted = st.form_submit_button("Predict")

# 예측
if submitted:
    try:
        freq_val = float(frequency)
        imp_val = float(impedance)
        time_val = float(time)

        # 유효성 검사
        if not (10 <= freq_val <= 100000):
            st.warning("📛 Frequency는 10 ~ 100000 사이어야 합니다.")
        elif not (1000 <= imp_val <= 50000):
            st.warning("📛 Impedance는 1000 ~ 50000Ω 사이어야 합니다.")
        elif not (0 <= time_val <= 1440):
            st.warning("📛 Time은 0 ~ 1440 분 사이어야 합니다.")
        else:
            X_input = np.array([[freq_val, imp_val, time_val]])
            X_scaled = scaler_X.transform(X_input)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            st.success("✅ 예측 완료!")
            st.markdown(f"""
            - **Interpolated Surface area (m²/g)**: `{y_pred[0][0]:.2f}`
            - **Interpolated Pore diameter (nm)**: `{y_pred[0][1]:.2f}`
            - **Interpolated Pore Volume (cm³/g)**: `{y_pred[0][2]:.4f}`
            """)

    except ValueError:
        st.error("❗ 모든 입력 값은 숫자여야 합니다. 다시 확인해 주세요.")

