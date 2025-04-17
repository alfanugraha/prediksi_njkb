import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

df_ori = pd.read_csv("data/training.csv")
df = df_ori.dropna()

def prediction(input_df):
    progress_bar = st.sidebar.progress(0)

    # Prepare data for modeling
    X = df[['KENDARAAN', 'JENIS_KENDARAAN', 'MEREK', 'NEGARA', 'TAHUN_BUAT', 'ODOMETER']]
    y = df['HARGA']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['KENDARAAN', 'JENIS_KENDARAAN', 'MEREK', 'NEGARA']),
            ('num', StandardScaler(), ['TAHUN_BUAT', 'ODOMETER'])
        ])

    # Results dictionary
    results = {
        'Model': [],
        'Training Time (s)': [],
        'MSE': [],
        'MAE': [],
        'R²': []
    }

    # Create pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # for i in range(1, 101):
    #     rf_pipeline.n_estimators = i
    #     rf_pipeline.fit(X_train, y_train)

    #     progress = i / 100
    #     progress_bar.progress(progress)
    #     time.sleep(0.05)

    # progress_bar.empty()

    rf_pipeline.fit(X_train, y_train)

    # Make predictions
    st.button("Re-run")
    return round(float(rf_pipeline.predict(input_df)[0]), 2)


# pick kendaraan
kendaraan_set = list(set(df_ori['KENDARAAN']))
kendaraan_selectbox = st.sidebar.selectbox("KENDARAAN", kendaraan_set)

selected_kendaraan_df = df_ori[df_ori["KENDARAAN"] == kendaraan_selectbox]

# jenis kendaraan 
jenis_kendaraan_set = list(set(selected_kendaraan_df['JENIS_KENDARAAN']))
jenis_kendaraan_selectbox = st.sidebar.selectbox("JENIS KENDARAAN", jenis_kendaraan_set)

# pick negara
negara_set = list(set(selected_kendaraan_df['NEGARA']))
negara_selectbox = st.sidebar.selectbox("NEGARA", negara_set)

# pick merk
merk_set = list(set(selected_kendaraan_df['MEREK']))
merk_selectbox = st.sidebar.selectbox("MEREK", merk_set)

# pick transmisi
# transmisi_set = list(set(selected_kendaraan_df['TRANSMISI']))
# transmisi_selectbox = st.sidebar.selectbox("TRANSMISI", negara_set)

# pick tahun
tahun_set = list(set(selected_kendaraan_df['TAHUN_BUAT']))
tahun_selectbox = st.sidebar.selectbox("TAHUN BUAT", tahun_set)

odometer = st.sidebar.number_input("ODOMETER",  step=1)

input_df = pd.DataFrame({
    'MEREK': [merk_selectbox],
    'KENDARAAN': [kendaraan_selectbox],
    'ODOMETER': [odometer],
    # 'TRANSMISI': [transmisi_selectbox],
    'JENIS_KENDARAAN': [jenis_kendaraan_selectbox],
    'TAHUN_BUAT': [tahun_selectbox],
    'NEGARA': [negara_selectbox]
})

st.markdown("# Prediksi Harga Pasar")
st.write(kendaraan_selectbox)
st.write(jenis_kendaraan_selectbox)
st.write(negara_selectbox)
st.write(merk_selectbox)
# st.write(transmisi_selectbox)
st.write(tahun_selectbox)
st.write(odometer)

res = prediction(input_df)
st.write(f"Prediksi Harga Pasar Rp. {res}")