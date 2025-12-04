from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
import logging

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Load Model & Scaler ----
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    logger.info("✅ Model and Scaler loaded successfully")
except FileNotFoundError as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    logger.error(
        "Make sure to run the notebook first to generate model.pkl and scaler.pkl")
    model = None
    scaler = None

# ---- Column names from ids_project notebook ----
COLUMN_NAMES = [
    'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
    'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate', 'exercise_angina',
    'oldpeak', 'slope', 'ca', 'thal', 'country'
]

app = Flask(__name__)


@app.route('/')
def home():
    """Render the home page"""
    return render_template('project.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form input"""

    if model is None or scaler is None:
        return render_template("result.html",
                               result="❌ Model not loaded. Please run the notebook first.",
                               color="red")

    try:
        # Extract input from form - supporting both old and new column names
        age = float(request.form.get('age', 0))
        sex = float(request.form.get('sex', 0))

        # chest_pain_type or cp
        chest_pain_type = float(request.form.get('chest_pain_type',
                                                 request.form.get('cp', 0)))

        # resting_bp or trestbps
        resting_bp = float(request.form.get('resting_bp',
                                            request.form.get('trestbps', 0)))

        # cholesterol or chol
        cholesterol = float(request.form.get('cholesterol',
                                             request.form.get('chol', 0)))

        # fasting_blood_sugar or fbs
        fasting_blood_sugar = float(request.form.get('fasting_blood_sugar',
                                                     request.form.get('fbs', 0)))

        # rest_ecg or restecg
        rest_ecg = float(request.form.get('rest_ecg',
                                          request.form.get('restecg', 0)))

        # max_heart_rate or thalach
        max_heart_rate = float(request.form.get('max_heart_rate',
                                                request.form.get('thalach', 0)))

        # exercise_angina or exang
        exercise_angina = float(request.form.get('exercise_angina',
                                                 request.form.get('exang', 0)))

        # oldpeak
        oldpeak = float(request.form.get('oldpeak', 0))

        # slope
        slope = float(request.form.get('slope', 0))

        # ca
        ca = float(request.form.get('ca', 0))

        # thal
        thal = float(request.form.get('thal', 0))

        # country
        country = float(request.form.get('country', 0))

        # Create dataframe with proper column names
        user_data = pd.DataFrame([[
            age, sex, chest_pain_type, resting_bp, cholesterol,
            fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina,
            oldpeak, slope, ca, thal, country
        ]], columns=COLUMN_NAMES)

        # Scale features
        user_data_scaled = scaler.transform(user_data)

        # Make prediction
        prediction = model.predict(user_data_scaled)[0]
        probability = model.predict_proba(user_data_scaled)[0]

        # Generate result
        if prediction == 1:
            result_text = f"⚠️ HIGH RISK of Heart Disease (Confidence: {probability[1]:.1%})"
            color = "red"
        else:
            result_text = f"✅ LOW RISK of Heart Disease (Confidence: {probability[0]:.1%})"
            color = "green"

        return render_template("result.html", result=result_text, color=color)

    except ValueError as e:
        logger.error(f"Error parsing input: {e}")
        return render_template("result.html",
                               result="❌ Invalid input. Please enter valid numbers.",
                               color="red")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("result.html",
                               result=f"❌ Error during prediction: {str(e)}",
                               color="red")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions (for testing)"""

    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        # Extract features in correct order
        user_data = pd.DataFrame([[
            data.get('age'),
            data.get('sex'),
            data.get('chest_pain_type'),
            data.get('resting_bp'),
            data.get('cholesterol'),
            data.get('fasting_blood_sugar'),
            data.get('rest_ecg'),
            data.get('max_heart_rate'),
            data.get('exercise_angina'),
            data.get('oldpeak'),
            data.get('slope'),
            data.get('ca'),
            data.get('thal'),
            data.get('country')
        ]], columns=COLUMN_NAMES)

        # Scale and predict
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]
        probability = model.predict_proba(user_data_scaled)[0]

        return jsonify({
            'prediction': int(prediction),
            'risk': 'HIGH' if prediction == 1 else 'LOW',
            'confidence_no_risk': float(probability[0]),
            'confidence_high_risk': float(probability[1])
        })

    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
