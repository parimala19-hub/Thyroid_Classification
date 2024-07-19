# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:37:29 2024

@author: SATHVIK
"""
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and label encoder
model_path = r'C:\Users\SATHVIK\OneDrive\Desktop\smartinternz\Thyroid\training\thyroid_1_model.pkl'
encoder_path = r'C:\Users\SATHVIK\OneDrive\Desktop\smartinternz\Thyroid\training\label_encoder.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

@app.route('/')
def about():
    return render_template('home.html')

@app.route('/predict')
def home1():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        
        # Convert form inputs to appropriate types
        features = [
            float(form_data.get('age', 0)),
            1 if form_data.get('sex', 'male') == 'female' else 0,
            float(form_data.get('on_thyroxine', 0)),
            float(form_data.get('query_on_thyroxine', 0)),
            float(form_data.get('on_antithyroid_meds', 0)),
            float(form_data.get('sick', 0)),
            float(form_data.get('pregnant', 0)),
            float(form_data.get('thyroid_surgery', 0)),
            float(form_data.get('I131_treatment', 0)),
            float(form_data.get('query_hypothyroid', 0)),
            float(form_data.get('query_hyperthyroid', 0)),
            float(form_data.get('lithium', 0)),
            float(form_data.get('goitre', 0)),
            float(form_data.get('tumor', 0)),
            float(form_data.get('hypopituitary', 0)),
            float(form_data.get('psych', 0)),
            float(form_data.get('TSH', 0)),
            float(form_data.get('T3', 0)),
            float(form_data.get('TT4', 0)),
            float(form_data.get('T4U', 0)),
            float(form_data.get('FTI', 0)),
            float(form_data.get('TBG', 0))
        ]

        # Columns used during model training
        columns = [
            'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant',
            'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
            'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'
        ]
        
        # Create DataFrame for prediction
        features_df = pd.DataFrame([features], columns=columns)

        # Make prediction
        prediction = model.predict(features_df)
        
        # Decode prediction
        decoded_prediction = label_encoder.inverse_transform(prediction)[0]

        return render_template('submit.html', prediction=decoded_prediction)
    
    except Exception as e:
        app.logger.error(f"Error occurred during prediction: {e}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=False)
