# ML_life_expectancy_linearregressor
AI를 활용한 경제,보건,건강 상태와 기대수명과의 상관관계를 예측하여 시각화 (머신러닝)
제공해주신 GitHub 저장소의 내용을 바탕으로 구성한 `README.md` 파일 예시입니다. 이 프로젝트가 선형 회귀(Linear Regression)를 이용해 기대 수명을 예측하는 머신러닝 모델을 다루고 있는 점을 반영하여 작성했습니다.

---
브라우저로 앱확인: https://mllifeexpectancylinearregreapprgit-lw6st4djbzgjasdappnh8vd.streamlit.app/
# 🧬 Life Expectancy Prediction using Linear Regression

이 프로젝트는 국가별 다양한 보건, 경제, 사회적 요인 데이터를 바탕으로 **선형 회귀(Linear Regression)** 모델을 사용하여 **기대 수명(Life Expectancy)**을 예측하는 머신러닝 프로젝트입니다.

## 📌 프로젝트 개요

* **목표**: 기대 수명에 영향을 미치는 주요 요인을 분석하고, 이를 통해 기대 수명을 예측하는 회귀 모델을 구축합니다.
* **주요 기술 스택**:
* Python
* Pandas, NumPy (데이터 전처리)
* Matplotlib, Seaborn (데이터 시각화)
* Scikit-learn (머신러닝 모델링 및 평가)



## 📊 데이터셋 정보

본 프로젝트에서는 WHO(세계보건기구) 등에서 제공하는 기대 수명 데이터셋을 활용합니다. 주요 피처는 다음과 같습니다:

* **Target**: `Life expectancy` (기대 수명)
* **Features**:
* `Adult Mortality`: 성인 사망률
* `Infant deaths`: 영아 사망 수
* `Alcohol`: 알코올 소비량
* `GDP`: 국내 총생산
* `Schooling`: 교육 연수
* `BMI`: 체질량 지수 등



## 🛠 주요 작업 흐름

1. **데이터 탐색 (EDA)**: 데이터의 분포, 결측치 확인 및 변수 간의 상관관계 분석
2. **데이터 전처리**:
* 결측치 처리 (Imputation)
* 범주형 데이터 인코딩
* 데이터 스케일링 (Standardization/Normalization)


3. **모델 학습**: Scikit-learn의 `LinearRegression`을 사용한 모델 학습
4. **모델 평가**: R-squared, MSE(Mean Squared Error), RMSE 등을 통한 성능 평가
5. **결과 시각화**: 실제값 vs 예측값 비교 그래프 생성

## 🚀 시작하기

### 설치 방법

```bash
git clone https://github.com/yoonhj9622/ML_life_expectancy_linearregressor.git
cd ML_life_expectancy_linearregressor

```

### 필수 라이브러리 설치

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

### 실행 방법

Jupyter Notebook 또는 Python 스크립트를 실행하여 모델 학습 및 결과를 확인할 수 있습니다.

```bash
jupyter notebook life_expectancy_regression.ipynb

```

## 📈 분석 결과

* (예시: 특정 변수(예: 교육 수준, GDP)가 기대 수명과 강한 양의 상관관계를 보임)
* 모델의 결정 계수(R²) 점수 및 오차 분석 내용 작성

## 👤 작성자

* **Name**: yoonhj9622
* **GitHub**: [yoonhj9622](https://www.google.com/search?q=https://github.com/yoonhj9622)

---

### 💡 참고 사항

이 프로젝트는 교육 및 실습 목적으로 작성되었으며, 데이터의 특성에 따라 다중 공선성(Multicollinearity) 처리나 규제 모델(Ridge, Lasso) 적용 등 추가적인 최적화가 가능합니다.
