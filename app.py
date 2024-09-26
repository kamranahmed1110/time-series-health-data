from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained LSTM model
model = load_model('my_models.h5', compile=False)

# Load and fit scaler on the original data (same used in training)
data = pd.read_csv('health_time_series_indexed.csv')
scaler = MinMaxScaler()
scaler.fit(data)

# Preprocessing helper function to scale the input
def preprocess_input(input_data):
    # Check if the length of the input matches the expected feature count
    if len(input_data) != 5:
        raise ValueError("Input must have exactly 5 features.")

    input_data = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_data)
    return scaled_input

# Create sequences for LSTM from a single input
def create_sequence(data, seq_length=3):
    if len(data) < seq_length:
        raise ValueError("Input data is too short to create a sequence.")
    return np.array([data])

# Define a route to make predictions
@app.route('/predicts', methods=['POST'])
def predict():
    # Get the data from the POST request
    input_data = request.json.get('input')  # Ensure the key matches the incoming JSON

    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Preprocess and scale the input
        scaled_input = preprocess_input(input_data)
        
        # Create a sequence from the scaled input
        sequence = create_sequence(scaled_input[0], seq_length=3)

        # Reshape for LSTM [samples, time steps, features]
        sequence = sequence.reshape((1, sequence.shape[1], sequence.shape[2]))

        # Make prediction
        prediction = model.predict(sequence)

        # Inverse scaling to return original data range
        predicted_values = scaler.inverse_transform(prediction)

        # Return the prediction
        return jsonify({'prediction': predicted_values.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define the main route for testing
@app.route('/')
def index():
    return "Welcome to the Health Prediction API. Use /predict endpoint to make predictions."

if __name__ == '__main__':
    app.run(debug=True)
