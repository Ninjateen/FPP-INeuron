import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("Data storage","model.pkl")
            preprocessor_path=os.path.join('Data storage','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( self,
        #Airline: str,
        Source: str,
        Destination: str,
        Total_Stops: str,
        Date_of_Journey: str,
        ):

        self.Source = Source

        self.Destination = Destination

        self.Total_Stops = Total_Stops

        self.Date_of_Journey = Date_of_Journey

        #self.Airline = Airline


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Total_Stops": [self.Total_Stops],
                "Date_of_Journey": [self.Date_of_Journey],
                #"Airline": [self.Airline],
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
