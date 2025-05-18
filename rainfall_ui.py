import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import math

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor

MODEL_PATH = "rainfall_model.pkl"
MONTH_ENCODER_PATH = "month_encoder.pkl"
SUBDIVISION_ENCODER_PATH = "subdivision_encoder.pkl"

@st.cache_data
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(MONTH_ENCODER_PATH) and os.path.exists(SUBDIVISION_ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le_month = joblib.load(MONTH_ENCODER_PATH)
        le_subdivision = joblib.load(SUBDIVISION_ENCODER_PATH)
    else:
        df = pd.read_csv('rainfall in india 1901-2015.csv')
        df.dropna(inplace=True)
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        df_melted = df.melt(id_vars=['SUBDIVISION', 'YEAR'], value_vars=months,
                            var_name='MONTH', value_name='RAINFALL')

        le_month = LabelEncoder()
        le_subdivision = LabelEncoder()
        df_melted['MONTH_ENC'] = le_month.fit_transform(df_melted['MONTH'])
        df_melted['SUBDIVISION_ENC'] = le_subdivision.fit_transform(df_melted['SUBDIVISION'])

        X = df_melted[['MONTH_ENC', 'SUBDIVISION_ENC', 'YEAR']]
        y = df_melted['RAINFALL']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        LR = LinearRegression()
        random_forest_model = RandomForestRegressor(random_state=42)
        svm_regr = SVR()
        xgb = XGBRegressor(random_state=42)

        model = StackingRegressor(
            regressors=(LR, random_forest_model, svm_regr),
            meta_regressor=xgb,
            use_features_in_secondary=True
        )

        model.fit(X_train, y_train)

        # Save model and encoders
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le_month, MONTH_ENCODER_PATH)
        joblib.dump(le_subdivision, SUBDIVISION_ENCODER_PATH)

    return model, le_month, le_subdivision

# Load model
model, le_month, le_subdivision = load_model()

# UI
st.title("Rainfall Prediction App 🌧️")
st.markdown("Predict rainfall based on month, region, and year.")

month = st.selectbox("Select Month", list(le_month.classes_))
subdivision = st.selectbox("Select Subdivision", list(le_subdivision.classes_))
year = st.number_input("Enter Year", min_value=2025, max_value=2100, step=1)

if st.button("Predict Rainfall"):
    month_enc = le_month.transform([month])[0]
    sub_enc = le_subdivision.transform([subdivision])[0]
    input_vector = np.array([[month_enc, sub_enc, year]])
    prediction = model.predict(input_vector)[0]
    st.success(f"Expected rainfall in **{subdivision}** during **{month} {year}** is **{prediction:.2f} mm**.")
