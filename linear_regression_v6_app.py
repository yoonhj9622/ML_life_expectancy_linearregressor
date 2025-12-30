import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# 1. 모델 및 객체 로드
# ================================
@st.cache_resource
def load_linear_pack():
    model = joblib.load("./models1/linear_model.pkl")
    scaler = joblib.load("./models1/scaler.pkl")
    feature_columns = joblib.load("./models1/feature_columns.pkl")
    return model, scaler, feature_columns

try:
    model, scaler, FEATURE_COLUMNS = load_linear_pack()
except FileNotFoundError:
    st.error("모델 파일(pkl)을 찾을 수 없습니다. 먼저 학습 코드를 실행해 주세요.")
    st.stop()

# ================================
# 2. 페이지 설정
# ================================
st.set_page_config(
    page_title="Life Expectancy Predictor (Linear Regression)",
    layout="centered"
)
st.title("기대 수명 예측 서비스 (Linear Regression)")
st.caption("선형 회귀 모델을 활용한 기대 수명 분석 서비스")
st.markdown("---")

# ================================
# 3. UI 레이아웃
# ================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("기본 정보 및 경제")
    status = st.selectbox("국가 상태 (Status)", ["Developing", "Developed"])
    income = st.slider("소득 자원 구성", 0.0, 1.0, 0.6, 0.01)
    schooling = st.slider("교육 기간", 0.0, 20.0, 12.0, 0.5)
    gdp = st.slider("1인당 GDP", 0, 100000, 5000, 100)
    expenditure = st.slider("의료비 지출 비중 (%)", 0.0, 20.0, 5.0, 0.1)
    total_exp = st.slider("총 의료비 지출 (%)", 0.0, 15.0, 6.0, 0.1)

with col2:
    st.subheader("건강 및 사망 지표")
    adult_mortality = st.slider("성인 사망률", 0, 1000, 150)
    under_five_deaths = st.slider("5세 미만 사망수", 0, 2500, 50)
    hiv = st.slider("HIV/AIDS 발생률", 0.0, 50.0, 0.1, 0.1)
    bmi = st.slider("체질량지수 (BMI)", 10.0, 60.0, 25.0, 0.1)
    alcohol = st.slider("알코올 소비량", 0.0, 20.0, 4.0, 0.1)
    thinness = st.slider("10대 마름 정도 (%)", 0.0, 30.0, 5.0, 0.1)

st.subheader("🛡️ 예방접종률 (%)")
c1, c2, c3 = st.columns(3)
with c1:
    polio = st.slider("소아마비", 0, 100, 80)
with c2:
    diphtheria = st.slider("디프테리아", 0, 100, 80)
with c3:
    hepatitis = st.slider("B형 간염", 0, 100, 80)

measles = st.slider("홍역 발생 건수", 0, 10000, 500)

# ================================
# 4. 예측 처리
# ================================
st.markdown("---")

if st.button("🔍 기대 수명 예측하기", use_container_width=True):
    # 1️사용자 입력 구성
    input_dict = {
        "Adult Mortality": float(adult_mortality),
        "Alcohol": float(alcohol),
        "percentage expenditure": float(expenditure),
        "Hepatitis B": float(hepatitis),
        "Measles": float(measles),
        "BMI": float(bmi),
        "under-five deaths": float(under_five_deaths),
        "Polio": float(polio),
        "Total expenditure": float(total_exp),
        "Diphtheria": float(diphtheria),
        "HIV/AIDS": float(hiv),
        "GDP": float(gdp),
        "thinness  1-19 years": float(thinness), # 공백 두 개 확인 필요
        "Income composition of resources": float(income),
        "Schooling": float(schooling)
    }

    # 2️ 모델 입력 프레임 생성
    final_input = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLUMNS)

    # 3️ 데이터 채우기
    for col, value in input_dict.items():
        if col in final_input.columns:
            final_input[col] = value

    # 4️Status 원-핫 인코딩 반영
    if "Status_Developing" in FEATURE_COLUMNS:
        final_input["Status_Developing"] = 1 if status == "Developing" else 0
    elif "Status_Developed" in FEATURE_COLUMNS:
        final_input["Status_Developed"] = 1 if status == "Developed" else 0

    # 5️스케일링 및 예측 (로그 역변환 포함)
    scaled_data = scaler.transform(final_input)
    log_prediction = model.predict(scaled_data)[0]
    prediction = np.expm1(log_prediction)

    # 6결과 화면 출력
    st.balloons()
    st.markdown(
        f"""
        <div style="text-align:center; background-color:#f0f2f6;
                    padding:20px; border-radius:10px; border: 2px solid #1a237e;">
            <h2 style="color:#1a237e;">예측된 기대 수명 (Linear)</h2>
            <h1 style="color:#2e7d32; font-size:3.5rem;">
                {prediction:.2f} 년
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("🔎 모델 입력 데이터 확인"):
        st.dataframe(final_input)