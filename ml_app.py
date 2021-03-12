import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from PIL import Image,ImageFilter,ImageEnhance
import h5py
import tensorflow.keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle 
import joblib

def run_ml_app():
    st.subheader('Machine learning')

    gender = st.radio("성별 선택하세요" , ['남자','여자'])
    if gender == "남자":
        gender = 1
    else :
        gender = 0

    age = st.number_input("나이 입력", min_value = 0, max_value = 120)

    salary = st.number_input('연봉 입력' , min_value=0)
    
    debt = st.number_input('빚 입력' , min_value=0)

    worth = st.number_input('자산 입력' , min_value=0)


    #예측한다.


    model =tensorflow.keras.models.load_model('data/car_ai.h5')
    
    new_data = np.array([gender,age,salary,debt,worth])

    #new_data = np.array([0, 0.36  , 0.875 , 0.09547739, 0.48979592])
    
    new_data = new_data.reshape(1,-1)
    
    sc_X = joblib.load('data/sc_X.pkl')

    new_data = sc_X.transform(new_data)

    y_pred = model.predict(new_data)

    #st.write(predicted_data[0][0])
    
    sc_y = joblib.load('data/sc_y.pkl')

    y_pred_original = sc_y.inverse_transform( y_pred)

    if st.button("예측 결과 확인하기"):
        st.write(  "에측 결과입니다. {:,.1f} 달러의 차를 살 수 있습니다".format(y_pred_original[0][0])  )


#pip install scikit-learn==0.23.2


