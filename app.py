from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('model/booking_model.pkl')

# For demo, list the features your model expects (replace with actual features from train_model.py)
feature_columns = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
    'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

# Home route - shows a simple welcome message or landing page
@app.route('/')
def home():
    return "<h1>Welcome to Airline Booking Prediction API</h1><p>Use the /predict endpoint to get predictions.</p>"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON data from frontend
    
    # Validate data keys
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
    
    # Create dataframe from input to match model features
    try:
        input_df = pd.DataFrame([data], columns=feature_columns)
    except Exception as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    
    # Make prediction
    try:
        pred_encoded = model.predict(input_df)[0]
        # If you encoded target using LabelEncoder, decode back to label
        # For example, satisfaction: 0 = dissatisfied, 1 = neutral, 2 = satisfied
        target_map = {0: "dissatisfied", 1: "neutral or dissatisfied", 2: "satisfied"}
        prediction = target_map.get(pred_encoded, "Unknown")
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
