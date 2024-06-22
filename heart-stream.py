import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul aplikasi web
st.title('Aplikasi Prediksi Penyakit Jantung')
st.write('Isi formulir di bawah ini untuk memprediksi apakah Anda atau seseorang terkena penyakit jantung.')

# Fungsi untuk memuat model
def load_model():
    with open('finalized_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

# Tampilan form input
st.header('Formulir Input')

with st.form(key='heart_disease_form'):
    age = st.number_input('Umur', 1, 120, step=1)
    sex = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: 'Wanita' if x == 0 else 'Pria')
    chest_pain_type = st.selectbox('Tipe Nyeri Dada', [0, 1, 2], format_func=lambda x: 'ATA' if x == 0 else 'NAP' if x == 1 else 'ASY')
    resting_bp = st.number_input('Tekanan Darah Istirahat', 0, 200, step=1)
    cholesterol = st.number_input('Kolesterol', 0, 600, step=1)
    fasting_bs = st.selectbox('Gula Darah Puasa', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
    resting_ecg = st.selectbox('Resting ECG', [0, 1], format_func=lambda x: 'Normal' if x == 0 else 'ST')
    max_hr = st.number_input('Detak Jantung Maksimum', 0, 250, step=1)
    exercise_angina = st.selectbox('Angina Olahraga', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
    oldpeak = st.number_input('Oldpeak', 0.0, 10.0, step=0.1)
    st_slope = st.selectbox('ST Slope', [0, 1, 2], format_func=lambda x: 'Up' if x == 0 else 'Flat' if x == 1 else 'Down')

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    input_data = (age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    st.header('Hasil Prediksi')
    if prediction[0] == 0:
        st.success('Pasien Tidak Terkena Penyakit Jantung')
    else:
        st.error('Pasien Terkena Penyakit Jantung')

# Gaya tambahan untuk tampilan yang lebih menarik
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    h1 {
        color: #003399;
    }
    h2 {
        color: #003399;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
