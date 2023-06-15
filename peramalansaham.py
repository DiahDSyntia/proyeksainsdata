import streamlit as st
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title('Prediksi Saham pada perusahaan PT ASTRA INTERNATIONAL TBK')
st.write("""
### Random Forest Regressor
"""
)

tab_titles = [
    "Data",
    "Prepocessing Data",
    "Modelling",
    "Implementasi",
    ]
tabs = st.tabs(tab_titles)


with tabs[0]:
    st.write("""
    Aplikasi ini dibuat untuk memprediksi saham yang akan terjadi di besok, minggu depan, atau tahun depan
    """
    )
    st.write("""
    Data yang kami pakai merupakan data yang diambil dari yahoo finance. Data ini memiliki 7 fitur yaitu Date, open, high, low, close, adj close, volume
    """
    )
    st.write("""
    Data ini adalah data yang tercatat dalam kurung waktu Jun 15 2022 - Jun 15 2023
    """
    )
    st.write("""
    Data ini bertipe numerik
    """
    )
    data = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/proyeksainsdata/main/ASII_new.JK.csv')
    st.write(data)

with tabs[1]:
    option = st.radio("Pilih opsi", ("Min Max Scaler", "Reduksi Dimensi"))

with tabs[2]:
    model=st.selectbox(
            'Metode Prediksi', ('K-Nearest Neighbor','Random Forest Regressor','Decision Tree Regressor'))
    st.write("""
    # AKURASI
    """
    )

with tabs[3]:
    st.write("""
    # PREDIKSI SAHAM
    """
    )
    # Memasukkan input bilangan menggunakan Streamlit
    number_input1 = st.number_input("Input 1", value=0.0, step=0.1)
    number_input2 = st.number_input("Input 2", value=0.0, step=0.1)
    number_input3 = st.number_input("Input 3", value=0.0, step=0.1)
    number_input4 = st.number_input("Input 4", value=0.0, step=0.1)

    # Menambahkan tombol prediksi
    predict_button = st.button('Prediksi')


