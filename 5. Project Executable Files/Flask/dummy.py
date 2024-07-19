# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:42:34 2024

@author: SATHVIK
"""
import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

def load_model(model_path):
    """Loads a pickled model from the specified path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        The loaded model object, or None if an error occurs.
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except (EOFError, pickle.UnpicklingError) as e:
        raise ValueError(f"Error loading model: Pickle file is corrupted or incomplete: {e}")
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return None

class DummyModel:
    def predict(self, X):
        # Just return a dummy prediction
        return [0]

class DummyLabelEncoder:
    def inverse_transform(self, X):
        # Just return a dummy categorical prediction
        return ["dummy_category"]

app = Flask(__name__)

# Load model with error handling
try:
    model = load_model('thyroid_model.pkl')
    le = load_model('label_encoder.pkl')
    
    if model is None or le is None:
        raise ValueError("One or more models could not be loaded. Check the paths and integrity of the model files.")
except ValueError as e:
    print(e)
    print("Using dummy models for testing purposes.")
    model = DummyModel()
    le = DummyLabelEncoder()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def predict():
    try:
        # Input data validation
        input_data = request.form.to_dict()
        required_fields = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
        if not all(field in input_data for field in required_fields):
            return jsonify({'error': 'Missing required input fields'}), 400

        inputs = [float(value) for value in input_data.values()]

        # Prediction
        x = pd.DataFrame([inputs], columns=['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'])
        prediction = model.predict(x)[0]
        categorical_prediction = le.inverse_transform([prediction])[0]

        return render_template('submit.html', numerical_prediction=prediction, categorical_prediction=categorical_prediction)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
