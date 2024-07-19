import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load models and scalers
best_model = joblib.load('best_model.pkl')
X_scaler = joblib.load('X_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

# Load reference columns
smoothed_df_savgol = pd.read_csv('smoothed_data_lowess_savgol.csv')
output_columns = pd.read_csv('reference_results.csv').columns

# Function to predict soil properties
def predict_soil_properties(new_data):
    new_data_scaled = X_scaler.transform(new_data)
    predictions_scaled = best_model.predict(new_data_scaled)
    predictions = y_scaler.inverse_transform(predictions_scaled)
    predictions_df = pd.DataFrame(predictions, columns=output_columns)
    return predictions_df

# Function to visualize results
def visualize_results(y_test, y_pred, output_columns):
    bar_width = 0.35
    for i, col in enumerate(output_columns):
        st.write(f'Actual vs Predicted - {col}')
        indices = np.arange(len(y_test))
        plt.figure(figsize=(10, 6))
        plt.bar(indices - bar_width/2, y_test[:, i], bar_width, alpha=0.6, label='Actual')
        plt.bar(indices + bar_width/2, y_pred[:, i], bar_width, alpha=0.6, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

# Streamlit interface
st.title("Soil Properties Prediction from Wavelength Data")

st.write("Enter the values for the new sample:")
input_data = []
for col in smoothed_df_savgol.columns:
    value = st.number_input(f'{col}', value=0.0, format="%.2f")
    input_data.append(value)

if st.button('Predict'):
    new_sample_df = pd.DataFrame([input_data], columns=smoothed_df_savgol.columns)
    predicted_properties = predict_soil_properties(new_sample_df)
    st.write("Predicted Soil Properties:")
    st.write(predicted_properties)

    # Load test data for visualization (this part of the data should be loaded beforehand)
    combined_df = pd.read_csv('combined_data.csv')
    X_test = combined_df.iloc[:, :smoothed_df_savgol.shape[1]]  # Adjust columns selection based on your data structure
    y_test = combined_df.iloc[:, smoothed_df_savgol.shape[1]:]  # Adjust columns selection based on your data structure

    y_test_scaled = y_scaler.inverse_transform(y_test)
    y_pred_test = best_model.predict(X_scaler.transform(X_test))
    y_pred_test_scaled = y_scaler.inverse_transform(y_pred_test)

    visualize_results(y_test_scaled, y_pred_test_scaled, output_columns)
