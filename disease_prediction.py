# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:56:05 2024

@author: DHRUBAJYOTI
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open(r"./trained_model.sav", "rb"))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
def main():
    #giving a title 
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    Pregnancies= st.text_input('Number of pregnancies')
    Glucose= st.text_input('Glucpse Lvl')
    BloodPressure= st.text_input('BP Lvl')
    SkinThickness= st.text_input('Thickness of Skin')
    Insulin= st.text_input('Insulin Lvl')
    BMI= st.text_input('BMI')
    DiabetesPedigreeFunction= st.text_input('Diabetes Pedigree function Value')
    Age= st.text_input('Age')
    
    #code for prediction
    diagnosis= ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis= diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)    
    
if __name__=='__main__':
    main() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
