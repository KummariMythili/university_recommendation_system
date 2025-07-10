from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/fine_tune.pkl')
scaler = joblib.load('model/scaler.pkl')  # ✅ Load the scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gre_score = float(request.form['gre_score'])
        toefl_score = float(request.form['toefl_score'])
        university_rating = float(request.form['university_rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        research = float(request.form['research'])

        # Prepare DataFrame with correct feature names
        feature_names = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
        input_data = pd.DataFrame([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]], columns=feature_names)

        # ✅ Apply the saved scaler
        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_input)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted Chance of Admit: {prediction}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
