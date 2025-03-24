import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load artifacts
with open('model.pkl', 'rb') as f:
    model = pkl.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pkl.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pkl.load(f)

# Helper functions
def preprocess_input(input_df):
    # Process categorical features
    cat_cols = input_df.select_dtypes(exclude=np.number).columns
    encoded = pd.DataFrame(encoder.transform(input_df[cat_cols]))
    
    # Process numerical features
    num_cols = input_df.select_dtypes(include=np.number).columns
    scaled = pd.DataFrame(scaler.transform(input_df[num_cols]), columns=num_cols)
    
    return pd.concat([encoded, scaled], axis=1)

# Streamlit App
st.title('Obesity Prediction App')

# 1. Show raw data
if st.checkbox('Show raw data'):
    raw_data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    st.dataframe(raw_data)

# 2. Data Visualization
if st.checkbox('Show data visualization'):
    # Add your visualization code here
    pass

# Input form
st.sidebar.header('User Input Features')

# Numerical inputs
age = st.sidebar.slider('Age', 10, 100, 25)
height = st.sidebar.slider('Height (m)', 1.0, 2.5, 1.7)
weight = st.sidebar.slider('Weight (kg)', 30, 200, 70)

# Categorical inputs
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
family_history = st.sidebar.selectbox('Family History', ['yes', 'no'])
favc = st.sidebar.selectbox('FAVC', ['yes', 'no'])
caec = st.sidebar.selectbox('CAEC', ['Sometimes', 'Frequently', 'Always'])

# Create input dataframe
input_data = pd.DataFrame([[gender, age, height, weight, family_history, favc, 2, 3, caec]],
                         columns=['Gender', 'Age', 'Height', 'Weight', 
                                 'family_history_with_overweight', 'FAVC',
                                 'FCVC', 'NCP', 'CAEC'])

# 5. Show input data
st.subheader('User Input Data')
st.dataframe(input_data)

# Preprocess and predict
processed_input = preprocess_input(input_data)
probs = model.predict_proba(processed_input)
prediction = model.predict(processed_input)

# 6. Show probabilities
st.subheader('Prediction Probabilities')
classes = label_encoder.classes_
prob_df = pd.DataFrame(probs, columns=classes)
st.dataframe(prob_df)

# 7. Show final prediction
st.subheader('Final Prediction')
st.write(f'Predicted class: {label_encoder.inverse_transform(prediction)[0]}')
