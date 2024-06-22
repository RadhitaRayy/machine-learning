import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Judul aplikasi web
st.title('Aplikasi Prediksi Penyakit Jantung')

# Sidebar untuk navigasi
st.sidebar.title('Navigasi')
options = st.sidebar.selectbox('Pilih halaman:', ['Unggah Data', 'EDA', 'Pelatihan Model', 'Prediksi'])

# Inisialisasi variabel uploaded_file
uploaded_file = None

# Unggah Data
if options == 'Unggah Data':
    st.header('Unggah Data CSV Anda')
    uploaded_file = st.file_uploader('Pilih file', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

# EDA
if options == 'EDA':
    st.header('Analisis Data Eksploratif (EDA)')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()
        
        # Pranaproses Data
        df['sex'] = df['sex'].map({'F': 0, 'M': 1})
        df['chestpaintype'] = df['chestpaintype'].map({'ATA': 0, 'NAP': 1, 'ASY': 2})
        df['restingecg'] = df['restingecg'].map({'Normal': 0, 'ST': 1})
        df['exerciseangina'] = df['exerciseangina'].map({'N': 0, 'Y': 1})
        df['st_slope'] = df['st_slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
        df.dropna(subset=['chestpaintype', 'restingecg', 'st_slope'], inplace=True)
        df['chestpaintype'] = df['chestpaintype'].astype(int)
        df['restingecg'] = df['restingecg'].astype(int)
        df['st_slope'] = df['st_slope'].astype(int)

        # Heatmap
        st.subheader('Heatmap')
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        st.pyplot(plt)

        # Distribusi HeartDisease
        st.subheader('Distribusi Penyakit Jantung')
        plt.figure(figsize=(8, 5))
        sns.countplot(x='heartdisease', data=df)
        st.pyplot(plt)

        # Boxplot
        st.subheader('Boxplot')
        num_columns = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(num_columns):
            plt.subplot(2, 3, i + 1)
            sns.boxplot(x='heartdisease', y=col, data=df)
            plt.title(f'Distribusi {col} Berdasarkan Penyakit Jantung')
        plt.tight_layout()
        st.pyplot(plt)

        # Pairplot
        st.subheader('Pairplot')
        sns.pairplot(df, hue='heartdisease', vars=num_columns)
        st.pyplot(plt)
    else:
        st.write("Silakan unggah file CSV terlebih dahulu di halaman 'Unggah Data'.")

# Pelatihan Model
if options == 'Pelatihan Model':
    st.header('Pelatihan Model')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        # Pranaproses Data
        df['sex'] = df['sex'].map({'F': 0, 'M': 1})
        df['chestpaintype'] = df['chestpaintype'].map({'ATA': 0, 'NAP': 1, 'ASY': 2})
        df['restingecg'] = df['restingecg'].map({'Normal': 0, 'ST': 1})
        df['exerciseangina'] = df['exerciseangina'].map({'N': 0, 'Y': 1})
        df['st_slope'] = df['st_slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
        df.dropna(subset=['chestpaintype', 'restingecg', 'st_slope'], inplace=True)
        df['chestpaintype'] = df['chestpaintype'].astype(int)
        df['restingecg'] = df['restingecg'].astype(int)
        df['st_slope'] = df['st_slope'].astype(int)

        x = df.drop(columns='heartdisease', axis=1)
        y = df['heartdisease']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

        model = LogisticRegression()
        model.fit(x_train, y_train)

        x_train_prediction = model.predict(x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, y_train)

        x_test_prediction = model.predict(x_test)
        test_data_accuracy = accuracy_score(x_test_prediction, y_test)

        st.write(f'Akurasi data training: {training_data_accuracy}')
        st.write(f'Akurasi data testing: {test_data_accuracy}')

        # Simpan model
        with open('finalized_model.sav', 'wb') as model_file:
            pickle.dump(model, model_file)
    else:
        st.write("Silakan unggah file CSV terlebih dahulu di halaman 'Unggah Data'.")

# Prediksi
if options == 'Prediksi':
    st.header('Prediksi')
    if uploaded_file is not None:
        model_file = 'finalized_model.sav'
        if model_file is not None:
            model = pickle.load(open(model_file, 'rb'))

            age = st.number_input('Umur', 1, 120, step=1)
            sex = st.selectbox('Jenis Kelamin', [0, 1])
            chest_pain_type = st.selectbox('Tipe Nyeri Dada', [0, 1, 2])
            resting_bp = st.number_input('Tekanan Darah Istirahat', 0, 200, step=1)
            cholesterol = st.number_input('Kolesterol', 0, 600, step=1)
            fasting_bs = st.selectbox('Gula Darah Puasa', [0, 1])
            resting_ecg = st.selectbox('Resting ECG', [0, 1])
            max_hr = st.number_input('Detak Jantung Maksimum', 0, 250, step=1)
            exercise_angina = st.selectbox('Angina Olahraga', [0, 1])
            oldpeak = st.number_input('Oldpeak', 0.0, 10.0, step=0.1)
            st_slope = st.selectbox('ST Slope', [0, 1, 2])

            input_data = (age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            prediction = model.predict(input_data_reshaped)

            if prediction[0] == 0:
                st.write('Pasien Tidak Terkena Penyakit Jantung')
            else:
                st.write('Pasien Terkena Penyakit Jantung')
    else:
        st.write("Silakan unggah file CSV terlebih dahulu di halaman 'Unggah Data'.")