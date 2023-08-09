import os
import pandas as pd
from flask import Flask, render_template, request
from mlopsProject.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        try:
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            reading_score = float(request.form.get('writing_score'))
            writing_score = float(request.form.get('reading_score'))

            if not gender or not race_ethnicity or not parental_level_of_education or not lunch or not test_preparation_course:
                return render_template('home.html', error_message="Please fill out all fields.")

            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_data_frame()
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            if results:
                return render_template('home.html', results=results[0])
            else:
                return render_template('home.html', error_message="Prediction failed.")
        except Exception as e:
            return render_template('home.html', error_message="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
