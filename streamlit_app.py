import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_artifacts():
    with open('rf_model.pkl', 'rb') as f:
        model = pkl.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pkl.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pkl.load(f)
    with open('target_mapping.pkl', 'rb') as f:
        target_mapping = pkl.load(f)
    return model, scaler, encoder, target_mapping

# Load artifacts
model, scaler, encoder, target_mapping = load_artifacts()

# Reverse mapping
reverse_target_mapping = {v: k for k, v in target_mapping.items()}

# Load raw data
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

st.title("Obesity Prediction App")

# 1. Menampilkan raw data
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# 2. Data Visualization
st.subheader("Data Visualization")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x='NObeyesdad', data=data, order=data['NObeyesdad'].value_counts().index, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 3 & 4. Input Data User
st.subheader("Input Your Data")
num_features = data.select_dtypes(include=np.number).columns.tolist()
cat_features = data.select_dtypes(exclude=np.number).columns.tolist()

user_data = {}
for col in num_features:
    user_data[col] = st.slider(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))

for col in cat_features:
    user_data[col] = st.selectbox(f"{col}", data[col].unique())

# Convert to DataFrame
user_df = pd.DataFrame([user_data])
st.write("### User Input:", user_df)

# 5. Preprocessing user data
user_df_encoded = pd.DataFrame(encoder.transform(user_df[cat_features]), columns=encoder.get_feature_names_out())
user_df_scaled = pd.DataFrame(scaler.transform(user_df[num_features]), columns=num_features)
user_df_processed = pd.concat([user_df_encoded, user_df_scaled], axis=1)

# 6. Prediksi
prediction_probs = model.predict_proba(user_df_processed)[0]
predicted_class = model.predict(user_df_processed)[0]

st.subheader("Prediction Results")
st.write("### Class Probabilities:")
prob_df = pd.DataFrame({'Class': [reverse_target_mapping[i] for i in range(len(prediction_probs))], 'Probability': prediction_probs})
st.write(prob_df)

st.write(f"### Final Prediction: {reverse_target_mapping[predicted_class]}")
