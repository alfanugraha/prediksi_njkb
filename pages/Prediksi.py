import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

pkl_file = 'data/datamart.pkl'

def check_pickle_file():    
    if os.path.isfile(pkl_file):
        return True            
    else:
        return False

def populate_selectbox(df):
    df_ori = df

    # pick kendaraan
    kendaraan_set = list(set(df_ori['KENDARAAN']))
    kendaraan_selectbox = st.sidebar.selectbox("KENDARAAN", kendaraan_set)

    selected_kendaraan_df = df_ori[df_ori["KENDARAAN"] == kendaraan_selectbox]

    # jenis kendaraan 
    jenis_kendaraan_set = list(set(selected_kendaraan_df['JENIS_KENDARAAN']))
    jenis_kendaraan_set2 = [item for item in jenis_kendaraan_set if item is not None]
    jenis_kendaraan_selectbox = st.sidebar.selectbox("JENIS KENDARAAN", jenis_kendaraan_set2)

    selected_jeniskendaraan_df = selected_kendaraan_df[selected_kendaraan_df["JENIS_KENDARAAN"] == jenis_kendaraan_selectbox]

    # pick tipe kendaraaan
    tipe_set = list(set(selected_jeniskendaraan_df['TIPE_KENDARAAN']))
    tipe_set2 = [item for item in tipe_set if item is not None]
    tipe_selectbox = st.sidebar.selectbox("TIPE KENDARAAN", tipe_set2)

    selected_tipekendaraan_df = selected_jeniskendaraan_df[selected_jeniskendaraan_df["TIPE_KENDARAAN"] == tipe_selectbox]

    # pick merk
    merk_set = list(set(selected_tipekendaraan_df['MEREK']))
    merk_selectbox = st.sidebar.selectbox("MEREK", merk_set)

    selected_merk_df = selected_tipekendaraan_df[selected_tipekendaraan_df["MEREK"] == merk_selectbox]

    year_input = st.sidebar.selectbox('TAHUN BUAT', range(1990, 2025))
    odometer = st.sidebar.number_input("ODOMETER", value=10000, step=1)

    single_df = pd.DataFrame({
        'KENDARAAN': [kendaraan_selectbox],
        'JENIS_KENDARAAN': [jenis_kendaraan_selectbox],
        'TIPE_KENDARAAN': [tipe_selectbox],
        'MEREK': [merk_selectbox],
        'TAHUN_BUAT': [year_input],
        'ODOMETER': [odometer]
    })

    st.markdown("# Prediksi Harga Pasar")
    return single_df

def training(train_df):
    df = train_df[['KENDARAAN', 'JENIS_KENDARAAN', 'TIPE_KENDARAAN', 'MEREK', 'TAHUN_BUAT', 'ODOMETER', 'HARGA']].dropna()

    # Prepare data for modeling
    X = df[['KENDARAAN', 'JENIS_KENDARAAN', 'TIPE_KENDARAAN', 'MEREK', 'TAHUN_BUAT', 'ODOMETER']]
    y = df['HARGA']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['KENDARAAN', 'JENIS_KENDARAAN', 'TIPE_KENDARAAN', 'MEREK']),
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

    with open('data/pipeline.pkl', 'wb') as f:  
        pickle.dump(rf_pipeline, f)

    st.button("Re-train")
    return rf_pipeline


def prediction(pipeline, input_df):
    return round(float(pipeline.predict(input_df)[0]), 2)

st.set_page_config(page_title="Prediksi", page_icon="📈",)

if check_pickle_file():
    with open(pkl_file, 'rb') as f:
        df_ori = pickle.load(f)

    df_input = populate_selectbox(df_ori)
    st.dataframe(df_input, hide_index=True)

    if st.sidebar.button('Train'):
        with st.spinner('Training model...'):
            rf_pipeline = training(df_ori)

    if st.sidebar.button('Prediksi'):
        with st.spinner('Prediksi harga...'):
            with open('data/pipeline.pkl', 'rb') as f:
                rf = pickle.load(f)
            res = prediction(rf, df_input)
        st.info(f"Prediksi Harga Pasar Rp. {res}")
else:
    st.error('File tidak ditemukan! Unduh data pada menu Database')