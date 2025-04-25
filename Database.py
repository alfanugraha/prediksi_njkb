import streamlit as st
import pandas as pd
import pickle
import cx_Oracle

from streamlit_extras.stateful_button import button

def check_oracle_connection(con):
    try:
        con.ping()
        return True
    except cx_Oracle.Error:
        return False

def run():
    st.set_page_config(page_title="Database", page_icon="📥")

    st.write("# Unduh Data")

    if button("Cek Koneksi DB", key="button1"): 
        username = st.secrets["db_user"] 
        password = st.secrets["db_pass"] 
        ip = st.secrets["ip"] 
        port = st.secrets["port"] 
        service_name = st.secrets["service_name"] 
        dsn_tns = cx_Oracle.makedsn(ip, port, service_name=service_name)
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
        
        if check_oracle_connection(connection):
            st.success('Koneksi sukses')

            if st.button("Unduh Datamart"):
                with st.spinner("Memuat..."):               
                    cursor = connection.cursor()
                    query = "SELECT MEREK, BODY_TYPE, HARGA, KENDARAAN, ODOMETER, TIPE_KENDARAAN, JENIS_KENDARAAN, TAHUN_BUAT FROM DATAMART_SCRAPPING_NJKB_V6"
                    cursor.execute(query)
                    columns = [col[0] for col in cursor.description]

                    # Fetch results
                    datamart = cursor.fetchall()
                    df_ori = pd.DataFrame(datamart, columns=columns)

                    with open('data/datamart.pkl', 'wb') as f:  
                        pickle.dump(df_ori, f)

                st.success("Data telah tersimpan.")
                connection.close()

        else:
            st.error('Gagal. Periksa koneksi!')

if __name__ == "__main__":
    run()