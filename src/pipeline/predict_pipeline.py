import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("Data storage", "model.pkl")
            preprocessor_path = os.path.join('Data storage', 'preprocessor.pkl')
            print("Before Loading")
            model = utils.load_object(file_path=model_path)
            preprocessor = utils.load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise exception.CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Total_Stops: str,
                 month_of_journey: str,
                 day_of_journey: str,
                 Dep_Hour: str,
                 Dep_Min: str,
                 Duration_Total_Hour: str,
                 Arrival_Hour: str,
                 Arrival_Min: str,
                 Airline: str,
                 Source: str,
                 Destination: str,
                 Price: float):
        
        self.Total_Stops = Total_Stops
        self.month_of_journey = month_of_journey
        self.day_of_journey = day_of_journey
        self.Dep_Hour = Dep_Hour
        self.Dep_Min = Dep_Min
        self.Duration_Total_Hour = Duration_Total_Hour
        self.Arrival_Hour = Arrival_Hour
        self.Arrival_Min = Arrival_Min
        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Price = Price

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Total_Stops": [self.Total_Stops],
                "month_of_journey": [self.month_of_journey],
                "day_of_journey": [self.day_of_journey],
                "Dep_Hour": [self.Dep_Hour],
                "Dep_Min": [self.Dep_Min],
                "Duration_Total_Hour": [self.Duration_Total_Hour],
                "Arrival_Hour": [self.Arrival_Hour],
                "Arrival_Min": [self.Arrival_Min],
                "Airline_" + self.Airline.replace(" ", "_"): [1],
                "Source_" + self.Source.replace(" ", "_"): [1],
                "Destination_" + self.Destination.replace(" ", "_"): [1],
                "Price": [self.Price]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise exception.CustomException(e, sys)