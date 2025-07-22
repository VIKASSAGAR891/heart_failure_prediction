from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import gzip

app = Flask(__name__)

# Load model and scaler using gzip
with gzip.open('model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the expected input feature order
feature_columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                   'ejection_fraction', 'high_blood_pressure', 'platelets',
                   'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            input_data = [float(request.form[col]) for col in feature_columns]
            input_df = pd.DataFrame([input_data], columns=feature_columns)
            scaled_input = scaler.transform(input_df)

            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, probability=probability)


if __name__ == '__main__':
    app.run(debug=True)
