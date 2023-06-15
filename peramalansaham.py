import streamlit as st
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from numpy import array
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
import altair as alt
import pickle


st.title('Prediksi Saham pada perusahaan PT ASTRA INTERNATIONAL TBK')

tab_titles = [
    "Data",
    "Prepocessing Data",
    "Modelling",
    "Implementasi",
    ]
tabs = st.tabs(tab_titles)

st.sidebar.write("""
            Nama: Diah Dwi Syntia (200411100001)"""
            )
st.sidebar.write("""
            Nama: Rosita Dewi Lutfiyah (200411100002)
            """
            )


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
    df = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/proyeksainsdata/main/ASII_new.JK.csv')
    st.write(df)

with tabs[1]:
    # option = st.radio("Pilih opsi", ("Min Max Scaler", "Reduksi Dimensi"))
    data = df['Close']
    n = len(data)
    # membagi data menjadi 80% untuk data training dan 20% data testing
    sizeTrain = (round(n*0.8))
    data_train = pd.DataFrame(data[:sizeTrain])
    data_test = pd.DataFrame(data[sizeTrain:])
    st.write("""Berikut adalah prepocessing dari data saham PT ASTRA INTERNATIONAL TBK""")
    st.write("""Dilakukan Normalisasi Menggunakan MinMax Scaler""")
    min_ = st.checkbox('MinMax Scaler')
    mod = st.button("Cek")
    # melakukan normalisasi menggunakan minMaxScaler
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(data_train)
    # Mengaplikasikan MinMaxScaler pada data pengujian
    test_scaled = scaler.transform(data_test)
    # reshaped_data = data.reshape(-1, 1)
    train = pd.DataFrame(train_scaled, columns = ['data'])
    train = train['data']
    test = pd.DataFrame(test_scaled, columns = ['data'])
    test = test['data']
    if min_:
      if mod:
         st.write("Data Training MinMax Scaler")
         train
         st.write("Data Test MinMax Scaler")
         train
    def split_sequence(sequence, n_steps):
      X, y = list(), list()
      for i in range(len(sequence)):
         # find the end of this pattern
         end_ix = i + n_steps
         # check if we are beyond the sequence
         if end_ix > len(sequence)-1:
            break
         # gather input and output parts of the pattern
         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
         X.append(seq_x)
         y.append(seq_y)
      return array(X), array(y)
    #memanggil fungsi untuk data training
    df_X, df_Y = split_sequence(train, 4)
    x = pd.DataFrame(df_X, columns = ['xt-4','xt-3','xt-2','xt-1'])
    y = pd.DataFrame(df_Y, columns = ['xt'])
    dataset_train = pd.concat([x, y], axis=1)
    dataset_train.to_csv('data-train.csv', index=False)
    X_train = dataset_train.iloc[:, :4].values
    Y_train = dataset_train.iloc[:, -1].values
    #memanggil fungsi untuk data testing
    test_x, test_y = split_sequence(test, 4)
    x = pd.DataFrame(test_x, columns = ['xt-4','xt-3','xt-2','xt-1'])
    y = pd.DataFrame(test_y, columns = ['xt'])
    dataset_test = pd.concat([x, y], axis=1)
    dataset_test.to_csv('data-test.csv', index=False)
    X_test = dataset_test.iloc[:, :4].values
    Y_test = dataset_test.iloc[:, -1].values

with tabs[2]:
    def tuning(X_train,Y_train,X_test,Y_test,iterasi):
        hasil = 1
        iter = 0
        for i in range(1,iterasi):
            neigh = KNeighborsRegressor(n_neighbors=i)
            neigh = neigh.fit(X_train,Y_train)
            y_pred=neigh.predict(X_test)
            reshaped_data = y_pred.reshape(-1, 1)
            original_data = scaler.inverse_transform(reshaped_data)
            reshaped_datates = Y_test.reshape(-1, 1)
            actual_test = scaler.inverse_transform(reshaped_datates)
            akhir1 = pd.DataFrame(original_data)
            akhir = pd.DataFrame(actual_test)
            mape = mean_absolute_percentage_error(original_data, actual_test)
            if mape < hasil:
                hasil = mape
                iter = i
        return hasil, iter
    akr,iter = tuning(X_train,Y_train,X_test,Y_test,30)
    # Model knn
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X_train,Y_train)
    y_pred=neigh.predict(X_test)
    reshaped_data = y_pred.reshape(-1, 1)
    original_data = scaler.inverse_transform(reshaped_data)
    reshaped_datates = Y_test.reshape(-1, 1)
    actual_test = scaler.inverse_transform(reshaped_datates)
    akhir1 = pd.DataFrame(original_data)
    akhir1.to_csv('prediksi.csv', index=False)
    akhir = pd.DataFrame(actual_test)
    akhir.to_csv('aktual.csv', index=False)
    mape_knn = mean_absolute_percentage_error(original_data, actual_test)

    #Model Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)
    reshaped_data = y_pred.reshape(-1, 1)
    original_data = scaler.inverse_transform(reshaped_data)
    reshaped_datates = Y_test.reshape(-1, 1)
    actual_test = scaler.inverse_transform(reshaped_datates)
    akhir1 = pd.DataFrame(original_data)
    akhir1.to_csv('prediksi.csv', index=False)
    akhir = pd.DataFrame(actual_test)
    akhir.to_csv('aktual.csv', index=False)
    mape_rf = mean_absolute_percentage_error(original_data, actual_test)


    # Model dtr
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, Y_train)
    y_pred_dtr=regressor.predict(X_test)
    reshaped_data = y_pred_dtr.reshape(-1, 1)
    _original_data = scaler.inverse_transform(reshaped_data)
    reshaped_datates = Y_test.reshape(-1, 1)
    _actual_test = scaler.inverse_transform(reshaped_datates)
    mape_dtr = mean_absolute_percentage_error(_original_data, _actual_test)

    st.subheader("Ada beberapa pilihan model dibawah ini!")
    st.write("Pilih Model yang ingin di cek akurasinya")
    kn = st.checkbox('K-Nearest Neighbor')
    rf = st.checkbox('Random Forest Regressor')
    des = st.checkbox('Decision Tree')
    st.write("""
    # AKURASI
    """
    )
    mod = st.button("Modeling")


    if kn :
        if mod:
            st.write('Model KNN Menghasilkan Mape: {}'. format(mape_knn))
    if rf :
        if mod:
            st.write('Model KNN Menghasilkan Mape: {}'. format(mape_rf))
    if des :
        if mod:
            st.write("Model Decision Tree Menghasilkan Mape : {}" . format(mape_dtr))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
                'Nilai Mape' : [mape_knn,mape_rf,mape_dtr],
                'Nama Model' : ['KNN','Random Forest','Decision Tree']
        })
        bar_chart = alt.Chart(source).mark_bar().encode(
                y = 'Nilai Mape',
                x = 'Nama Model'
        )
        st.altair_chart(bar_chart,use_container_width=True)
    

with tabs[3]:
    st.write("""
    # PREDIKSI SAHAM
    """
    )
    #menyimpan model
    with open('knn','wb') as r:
        pickle.dump(neigh,r)
    with open('minmax','wb') as r:
        pickle.dump(scaler,r)
    
    input_1 = st.number_input('Masukkan Data 1')
    input_2 = st.number_input('Masukkan Data 2')
    input_3 = st.number_input('Masukkan Data 3')
    input_4 = st.number_input('Masukkan Data 4')

    def submit():
      # inputs = np.array([inputan])
      with open('knn', 'rb') as r:
         model = pickle.load(r)
      with open('minmax', 'rb') as r:
         minmax = pickle.load(r)
      data1 = minmax.transform([[input_1]])
      data2 = minmax.transform([[input_2]])
      data3 = minmax.transform([[input_3]])
      data4 = minmax.transform([[input_4]])

      X_pred = model.predict([[(data1[0][0]),(data2[0][0]),(data3[0][0]),(data4[0][0])]])
      t_data1= X_pred.reshape(-1, 1)
      original = minmax.inverse_transform(t_data1)
      hasil =f"Prediksi Hasil Peramalan Pada Harga Penutupan Saham PT Astra International Tbk adalah  : {original[0][0]}"
      st.success(hasil)

    all = st.button("Submit")
    if all :
        submit()


