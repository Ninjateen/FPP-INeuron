import sys
import os
import pandas as pd
sys.path.append('src')
sys.path.append("src/components")
from exception import CustomException
from logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("Data storage", "model.pkl")
            preprocessor_path = os.path.join('Data storage', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Total_Stops: str,
                 month_of_journey: float,
                 day_of_journey: float,
                 Dep_Hour: float,
                 Dep_Min: float,
                 Duration_Total_Hour: float,
                 Arrival_Hour: float,
                 Arrival_Min: float,
                 Airline_Air_Asia: float,
                 Airline_Air_India: float,
                 Airline_GoAir: float,
                 Airline_IndiGo: float,
                 Airline_Jet_Airways: float,
                 Airline_Jet_Airways_Business: float,
                 Airline_Multiple_carriers: float,
                 Airline_Multiple_carriers_Premium_economy: float,
                 Airline_SpiceJet: float,
                 Airline_Trujet: float,
                 Airline_Vistara: float,
                 Airline_Vistara_Premium_economy: float,
                 Source_Banglore: float,
                 Source_Chennai: float,
                 Source_Delhi: float,
                 Source_Kolkata: float,
                 Source_Mumbai: float,
                 Destination_Banglore: float,
                 Destination_Cochin: float,
                 Destination_Delhi: float,
                 Destination_Hyderabad: float,
                 Destination_Kolkata: float,
                 Destination_New_Delhi: float,
                 Price: float):
        
        self.Total_Stops = Total_Stops
        self.month_of_journey = month_of_journey
        self.day_of_journey = day_of_journey
        self.Dep_Hour = Dep_Hour
        self.Dep_Min = Dep_Min
        self.Duration_Total_Hour = Duration_Total_Hour
        self.Arrival_Hour = Arrival_Hour
        self.Arrival_Min = Arrival_Min
        self.Airline_Air_Asia = Airline_Air_Asia
        self.Airline_Air_India = Airline_Air_India
        self.Airline_GoAir = Airline_GoAir
        self.Airline_IndiGo = Airline_IndiGo
        self.Airline_Jet_Airways = Airline_Jet_Airways
        self.Airline_Jet_Airways_Business = Airline_Jet_Airways_Business
        self.Airline_Multiple_carriers = Airline_Multiple_carriers
        self.Airline_Multiple_carriers_Premium_economy = Airline_Multiple_carriers_Premium_economy
        self.Airline_SpiceJet = Airline_SpiceJet
        self.Airline_Trujet = Airline_Trujet
        self.Airline_Vistara = Airline_Vistara
        self.Airline_Vistara_Premium_economy = Airline_Vistara_Premium_economy
        self.Source_Banglore = Source_Banglore
        self.Source_Chennai = Source_Chennai
        self.Source_Delhi = Source_Delhi
        self.Source_Kolkata = Source_Kolkata
        self.Source_Mumbai = Source_Mumbai
        self.Destination_Banglore = Destination_Banglore
        self.Destination_Cochin = Destination_Cochin
        self.Destination_Delhi = Destination_Delhi
        self.Destination_Hyderabad = Destination_Hyderabad
        self.Destination_Kolkata = Destination_Kolkata
        self.Destination_New_Delhi = Destination_New_Delhi
        self.Price = Price

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {"Total_Stops": [self.Total_Stops],
                                      "month_of_journey": [self.month_of_journey],
                                      "day_of_journey": [self.day_of_journey],
                                      "Dep_Hour": [self.Dep_Hour],"Dep_Min": [self.Dep_Min],
                                      "Duration_Total_Hour": [self.Duration_Total_Hour],
                                      "Arrival_Hour": [self.Arrival_Hour],
                                      "Arrival_Min": [self.Arrival_Min],
                                      "Airline_Air_Asia": [self.Airline_Air_Asia],
                                      "Airline_Air_India": [self.Airline_Air_India],
                                      "Airline_GoAir": [self.Airline_GoAir],
                                      "Airline_IndiGo": [self.Airline_IndiGo],
                                      "Airline_Jet_Airways": [self.Airline_Jet_Airways],
                                      "Airline_Jet_Airways_Business": [self.Airline_Jet_Airways_Business],
                                      "Airline_Multiple_carriers": [self.Airline_Multiple_carriers],
                                      "Airline_Multiple_carriers_Premium_economy": [self.Airline_Multiple_carriers_Premium_economy],
                                      "Airline_SpiceJet": [self.Airline_SpiceJet],
                                      "Airline_Trujet": [self.Airline_Trujet],
                                      "Airline_Vistara": [self.Airline_Vistara],
                                      "Airline_Vistara_Premium_economy": [self.Airline_Vistara_Premium_economy],
                                      "Source_Banglore": [self.Source_Banglore],
                                      "Source_Chennai": [self.Source_Chennai],
                                      "Source_Delhi": [self.Source_Delhi],
                                      "Source_Kolkata": [self.Source_Kolkata],
                                      "Source_Mumbai": [self.Source_Mumbai],
                                      "Destination_Banglore": [self.Destination_Banglore],
                                      "Destination_Cochin": [self.Destination_Cochin],
                                      "Destination_Delhi": [self.Destination_Delhi],
                                      "Destination_Hyderabad": [self.Destination_Hyderabad],
                                      "Destination_Kolkata": [self.Destination_Kolkata],
                                      "Destination_New_Delhi": [self.Destination_New_Delhi],
                                      "Price": [self.Price]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
