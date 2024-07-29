import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_files():
    model = joblib.load('models/best_npk_model.pkl')
    scaler = joblib.load('models/npk_scaler.pkl')
    le = joblib.load('models/location_encoder.pkl')
    
    with open('models/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    with open('models/model_metrics.txt', 'r') as f:
        metrics = f.read()
    
    return model, scaler, le, feature_info, metrics

model, scaler, le, feature_info, metrics = load_model_files()

st.title('NPK Prediction App')

st.write("""
This app predicts the Nitrogen (N), Phosphorus (P), and Potassium (K) levels in compost based on available input features.
""")

input_data = {}

# Numeric features
for feature in feature_info['numeric_features']:
    if feature != 'Location': 
        input_data[feature] = st.number_input(f'{feature}', value=0.0, format="%.2f", step=0.05)
print(feature_info['numeric_features'])

# Categorical features (Location)
locations = le.classes_
selected_location = st.selectbox('Location', locations)
input_data['Location'] = le.transform([selected_location])[0]
print(locations)
print(input_data['Location'])

input_df = pd.DataFrame([input_data])

input_df = input_df[feature_info['features']]

numeric_features = [f for f in feature_info['numeric_features'] if f != 'Location']
input_df[numeric_features] = scaler.transform(input_df[numeric_features])


if st.button('Predict NPK Levels'):
    prediction = model.predict(input_df)
    
    st.subheader('Predicted NPK Levels')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    nutrients = feature_info['target_variables']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    bars = ax.bar(nutrients, prediction[0], color=colors)
    
    ax.set_ylabel('Predicted Level')
    ax.set_title('Predicted NPK Levels')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    st.pyplot(fig)
    
    for nutrient, value in zip(nutrients, prediction[0]):
        st.write(f"{nutrient}: {value:.2f}")
    
    st.write("""
    Note: These predictions are based on the current data available. 
    The actual NPK levels might be affected to other factors not considered in this model yet.
    """)

