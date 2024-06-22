import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul aplikasi web
st.title('Aplikasi Prediksi Penyakit Jantung')

# Fungsi untuk memuat model
def load_model():
    with open('finalized_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

# Input dari pengguna
age = st.number_input('Umur', 1, 120, step=1)
sex = st.selectbox('Jenis Kelamin (0: Wanita, 1: Pria)', [0, 1])
chest_pain_type = st.selectbox('Tipe Nyeri Dada (0: ATA, 1: NAP, 2: ASY)', [0, 1, 2])
resting_bp = st.number_input('Tekanan Darah Istirahat', 0, 200, step=1)
cholesterol = st.number_input('Kolesterol', 0, 600, step=1)
fasting_bs = st.selectbox('Gula Darah Puasa (0: Tidak, 1: Ya)', [0, 1])
resting_ecg = st.selectbox('Resting ECG (0: Normal, 1: ST)', [0, 1])
max_hr = st.number_input('Detak Jantung Maksimum', 0, 250, step=1)
exercise_angina = st.selectbox('Angina Olahraga (0: Tidak, 1: Ya)', [0, 1])
oldpeak = st.number_input('Oldpeak', 0.0, 10.0, step=0.1)
st_slope = st.selectbox('ST Slope (0: Up, 1: Flat, 2: Down)', [0, 1, 2])

input_data = (age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    st.write('Pasien Tidak Terkena Penyakit Jantung')
else:
    st.write('Pasien Terkena Penyakit Jantung')
