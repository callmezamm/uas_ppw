import streamlit as st
from streamlit_option_menu import option_menu
import joblib

# crawling
import requests
from bs4 import BeautifulSoup
import csv

# normalisasi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import re
import warnings
from nltk.stem import PorterStemmer

# VSM
from sklearn.feature_extraction.text import TfidfVectorizer

# LDA
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import os

# modeling
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib 
from joblib import load

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings('ignore')

with st.sidebar:
    selected = option_menu(
        menu_title= "Main Menu",
        options=["Home","Project"]
    )
if selected == "Home":
    st.header('PROJECT MACHINE LEARNING', divider='rainbow')
    st.write('Pada project ini akan membahas tentang proses crawling web Indo Times, melakukan normalisasi data abstrak, pemrposesan LDA untuk reduksi dimensi dan membuat model klasifikasi menggunakan DecisionTree')

    st.write("## **Crawling**")
    st.write("Crawling data adalah proses otomatis pengumpulan informasi atau data dari berbagai sumber di internet menggunakan program komputer atau robot yang disebut web crawler atau spider. Tujuan utama dari proses crawling data adalah untuk mengumpulkan informasi secara terstruktur dari berbagai situs web, mengindeksnya, dan membuatnya tersedia untuk analisis lebih lanjut, penelitian, atau penggunaan lainnya")
    st.write("## **Normalisasi**")
    st.write("Normalisasi data adalah proses pengelompokan atau penyesuaian data dalam suatu dataset sehingga mereka berada dalam rentang atau skala tertentu. Tujuan normalisasi adalah untuk memastikan bahwa data-data tersebut dapat dibandingkan atau digunakan dalam analisis tanpa ada bias yang muncul karena perbedaan dalam besaran atau satuan pengukuran.")
    st.write("## **Reduksi Dimensi LDA**")
    st.write("Reduksi dimensi dengan LDA (Linear Discriminant Analysis) adalah teknik analisis yang digunakan untuk mengurangi dimensi data dengan mempertimbangkan informasi diskriminatif antar kelas. LDA adalah metode yang umumnya digunakan dalam pengenalan pola dan klasifikasi, terutama dalam konteks pengolahan data dengan masalah klasifikasi.")
if selected == "Project":
    st.title("IMPLEMENTASI")

    st.header("Implementasi klasifikasi berita")
    inputan = st.text_area("Masukkan Berita")
    inputan = [inputan]
    # ======================== TF-IDF ==========================
    vertorizer = load('Model/tfidf_vectorizer.pkl')
    # ======================== LDA =============================
    lda = load('Model/best_lda_rf.pkl')
    # ======================== MODEL ===========================
    model = load('Model/best_model4_rf.pkl')


    if st.button("Prediksi"):
        ver_inp = vertorizer.transform(inputan)
        # st.write(ver_inp)
        lda_inp = lda.transform(ver_inp)
        predict_inp = model.predict(lda_inp)
        if predict_inp == "Olahraga":
            st.success("Hasil Prediksi dari Kalimat yang diinputkan menunjukkan : ")
            st.success("Olahraga")
        elif predict_inp == "Teknologi":
            st.success("Hasil Prediksi dari Kalimat yang diinputkan menunjukkan : ")
            st.success("Teknologi")
        else:
            st.success("Hasil Prediksi dari Kalimat yang diinputkan menunjukkan : ")
            st.success("Edukasi")
        