import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# 1. 모델 로드
# ==================================================
@st.cache_resource
def load_rf_pack():
    model = joblib.load("./models2/rf_model.pkl")
    scaler = joblib.load("./models2/scaler.pkl")
    feature_columns = joblib.load("./models2/feature_columns.pkl")
    return model, scaler, feature_columns

try:
    model, scaler, FEATURE_COLUMNS = load_rf_pack()
except FileNotFoundError:
    st.error("모델 파일이 없습니다. 먼저 학습 노트북을 실행하세요.")
    st.stop()

# ==================================================
# 2. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="Life Expectancy Predictor (Random Forest)",
    layout="centered"
)

st.title("🌍 기대 수명 예측 서비스 (Random Forest)")
st.caption("랜덤 포레스트 기반 기대 수명 예측")
st.markdown("---")

# ==================================================
# 3. UI 입력
# ==================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 경제·교육")
    status = st.selectbox("국가 상태", ["Developing", "Developed"])
    income = st.slider("소득 자원 구성", 0.0, 1.0, 0.6, 0.01)
    schooling = st.slider("교육 기간", 0.0, 20.0, 12.0, 0.5)
    gdp = st.slider("1인당 GDP", 0, 100000, 5000, 100)
    expenditure = st.slider("의료비 지출 비중", 0.0, 20.0, 5.0, 0.1)
    total_exp = st.slider("총 의료비 지출", 0.0, 15.0, 6.0, 0.1)

with col2:
    st.subheader("💉 건강 지표")
    adult_mortality = st.slider("성인 사망률", 0, 1000, 150)
    under_five = st.slider("5세 미만 사망수", 0, 2500, 50)
    hiv = st.slider("HIV/AIDS", 0.0, 50.0, 0.1, 0.1)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)
    alcohol = st.slider("알코올 소비량", 0.0, 20.0, 4.0, 0.1)
    thinness = st.slider("10대 마름 정도", 0.0, 30.0, 5.0, 0.1)

st.subheader("🛡 예방접종률")
c1, c2, c3 = st.columns(3)
with c1:
    polio = st.slider("소아마비", 0, 100, 80)
with c2:
    diphtheria = st.slider("디프테리아", 0, 100, 80)
with c3:
    hepatitis = st.slider("B형 간염", 0, 100, 80)

measles = st.slider("홍역 발생 건수", 0, 10000, 500)

# ==================================================
# 4. 예측
# ==================================================
st.markdown("---")

if st.button("🔍 기대 수명 예측하기", use_container_width=True):

    input_data = {
        "Adult Mortality": adult_mortality,
        "Alcohol": alcohol,
        "percentage expenditure": expenditure,
        "Hepatitis B": hepatitis,
        "Measles": measles,
        "BMI": bmi,
        "under-five deaths": under_five,
        "Polio": polio,
        "Total expenditure": total_exp,
        "Diphtheria": diphtheria,
        "HIV/AIDS": hiv,
        "GDP": gdp,
        "thinness  1-19 years": thinness,
        "Income composition of resources": income,
        "Schooling": schooling
    }

    final_input = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLUMNS)

    for col, val in input_data.items():
        if col in final_input.columns:
            final_input[col] = val

    if "Status_Developing" in FEATURE_COLUMNS:
        final_input["Status_Developing"] = 1 if status == "Developing" else 0

    scaled = scaler.transform(final_input)
    log_pred = model.predict(scaled)[0]
    prediction = np.expm1(log_pred)

    st.balloons()
    st.markdown(
        f"""
        <div style="text-align:center; background-color:#f0f2f6;
                    padding:20px; border-radius:10px; border:2px solid #2e7d32;">
            <h2>예측된 기대 수명</h2>
            <h1 style="color:#2e7d32; font-size:3.5rem;">
                {prediction:.2f} 년
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("🔎 입력 데이터 확인"):
        st.dataframe(final_input)
