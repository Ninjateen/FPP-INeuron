from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Source=request.form.get('Source'),
            Destination=request.form.get('Destination'),
            Total_Stops=request.form.get('Total_Stops'),
            day_of_journey=request.form.get('day_of_journey'),
            month_of_journey=request.form.get('month_of_journey'),
            Dep_Hour=request.form.get('Dep_Hour'),
            Dep_Min=request.form.get('Dep_Min'),
            Duration_Total_Hour=request.form.get('Duration_Total_Hour'),
            Arrival_Hour=request.form.get('Arrival_Hour'),
            Arrival_Min=request.form.get('Arrival_Min'),
            Airline_Air_Asia=request.form.get('Airline_Air_Asia'),
            Airline_Air_India=request.form.get('Airline_Air_India'),
            Airline_GoAir=request.form.get('Airline_GoAir'),
            Airline_IndiGo=request.form.get('Airline_IndiGo'),
            Airline_Jet_Airways=request.form.get('Airline_Jet_Airways'),
            Airline_Jet_Airways_Business=request.form.get('Airline_Jet_Airways_Business'),
            Airline_Multiple_carriers=request.form.get('Airline_Multiple_carriers'),
            Airline_Multiple_carriers_Premium_economy=request.form.get('Airline_Multiple_carriers_Premium_economy'),
            Airline_SpiceJet=request.form.get('Airline_SpiceJet'),
            Airline_Trujet=request.form.get('Airline_Trujet'),
            Airline_Vistara=request.form.get('Airline_Vistara'),
            Airline_Vistara_Premium_economy=request.form.get('Airline_Vistara_Premium_economy'),
            Source_Banglore=request.form.get('Source_Banglore'),
            Source_Chennai=request.form.get('Source_Chennai'),
            Source_Delhi=request.form.get('Source_Delhi'),
            Source_Kolkata=request.form.get('Source_Kolkata'),
            Source_Mumbai=request.form.get('Source_Mumbai'),
            Destination_Banglore=request.form.get('Destination_Banglore'),
            Destination_Cochin=request.form.get('Destination_Cochin'),
            Destination_Delhi=request.form.get('Destination_Delhi'),
            Destination_Hyderabad=request.form.get('Destination_Hyderabad'),
            Destination_Kolkata=request.form.get('Destination_Kolkata'),
            Destination_New_Delhi=request.form.get('Destination_New Delhi'),
            Price = 0 # We need to set a default value for Price
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)