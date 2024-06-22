import streamlit as st
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

# Penjelasan untuk masing-masing kolom input
st.header('Penjelasan Kolom Input:')
st.markdown('''
- **Umur:** Masukkan usia pasien dalam tahun.
- **Jenis Kelamin:** 0 = Wanita, 1 = Pria.
- **Tipe Nyeri Dada:** 0 = ATA, 1 = NAP, 2 = ASY.
- **Tekanan Darah Istirahat:** Tekanan darah dalam mmHg saat istirahat.
- **Kolesterol:** Kadar kolesterol dalam mg/dL.
- **Gula Darah Puasa:** 0 = Tidak, 1 = Ya.
- **Resting ECG:** 0 = Normal, 1 = ST.
- **Detak Jantung Maksimum:** Detak jantung maksimum yang dicapai.
- **Angina Olahraga:** 0 = Tidak, 1 = Ya.
- **Oldpeak:** Depresi ST yang disebabkan oleh olahraga relatif terhadap istirahat.
- **ST Slope:** 0 = Up, 1 = Flat, 2 = Down.
''')

# Tampilan form input dengan kolom
st.header('Formulir Input')

with st.form(key='heart_disease_form'):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Umur', 1, 120, step=1)
        sex = st.number_input('Jenis Kelamin (0: Wanita, 1: Pria)', 0, 1, step=1)
        chest_pain_type = st.number_input('Tipe Nyeri Dada (0: ATA, 1: NAP, 2: ASY)', 0, 2, step=1)
        resting_bp = st.number_input('Tekanan Darah Istirahat', 0, 200, step=1)
        cholesterol = st.number_input('Kolesterol', 0, 600, step=1)
    
    with col2:
        fasting_bs = st.number_input('Gula Darah Puasa (0: Tidak, 1: Ya)', 0, 1, step=1)
        resting_ecg = st.number_input('Resting ECG (0: Normal, 1: ST)', 0, 1, step=1)
        max_hr = st.number_input('Detak Jantung Maksimum', 0, 250, step=1)
        exercise_angina = st.number_input('Angina Olahraga (0: Tidak, 1: Ya)', 0, 1, step=1)
        oldpeak = st.number_input('Oldpeak', 0.0, 10.0, step=0.1)
    
    with col3:
        st_slope = st.number_input('ST Slope (0: Up, 1: Flat, 2: Down)', 0, 2, step=1)

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
