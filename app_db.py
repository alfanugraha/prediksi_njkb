import streamlit as st
import pandas as pd
import numpy as np
import cx_Oracle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

def prediction(train_df, input_df):
    progress_bar = st.sidebar.progress(0)
    df = train_df.dropna()

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

    # Make predictions
    st.button("Re-run")
    return round(float(rf_pipeline.predict(input_df)[0]), 2)


# df_ori = pd.read_csv("data/actual.csv")
username = st.secrets["db_user"] 
password = st.secrets["db_pass"] 
ip = st.secrets["ip"] 
port = st.secrets["port"] 
service_name = st.secrets["service_name"] 
dsn_tns = cx_Oracle.makedsn(ip, port, service_name=service_name)
connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
cursor = connection.cursor()
cursor.execute("SELECT MEREK, BODY_TYPE, HARGA, KENDARAAN, ODOMETER, TIPE_KENDARAAN, JENIS_KENDARAAN, TAHUN_BUAT FROM DATAMART_SCRAPPING_NJKB_V6")
columns = [col[0] for col in cursor.description]
# Fetch results
datamart = cursor.fetchall()
df_ori = pd.DataFrame(datamart, columns=columns)

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
odometer = st.sidebar.number_input("ODOMETER",  step=1)

single_df = pd.DataFrame({
    'KENDARAAN': [kendaraan_selectbox],
    'JENIS_KENDARAAN': [jenis_kendaraan_selectbox],
    'TIPE_KENDARAAN': [tipe_selectbox],
    'MEREK': [merk_selectbox],
    'TAHUN_BUAT': [year_input],
    'ODOMETER': [odometer]
})

st.markdown("# Prediksi Harga Pasar")
st.write(kendaraan_selectbox)
st.write(jenis_kendaraan_selectbox)
st.write(tipe_selectbox)
st.write(merk_selectbox)
st.write(year_input)
st.write(odometer)

res = prediction(df_ori, single_df)
st.write(f"Prediksi Harga Pasar Rp. {res}")