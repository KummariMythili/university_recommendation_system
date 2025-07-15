from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ✅ Load trained model and scaler
model = joblib.load('model/fine_tune.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Retrieve form data from HTML
        gre_score = float(request.form['gre_score'])
        toefl_score = float(request.form['toefl_score'])
        university_rating = float(request.form['university_rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        research = float(request.form['research'])

        # ✅ Create DataFrame with input features
        feature_names = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
        input_data = pd.DataFrame([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]],
                                  columns=feature_names)

        # ✅ Scale input using saved scaler
        scaled_input = scaler.transform(input_data)

        # ✅ Predict admission chance
        prediction = model.predict(scaled_input)[0]
        prediction = round(prediction * 100, 2)  # convert to percentage

        return render_template('index.html',
                               prediction_text=f"{prediction}%",
                               request=request)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"❌ Error: {e}",
                               request=request)

if __name__ == '__main__':
    app.run(debug=True)
