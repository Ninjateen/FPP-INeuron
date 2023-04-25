import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
sys.path.append('src')
sys.path.append("src\components")
from exception import CustomException
from logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Data storage',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns =  ['Duration_Total_Hour', 'Airline_Jet Airways', 'day_of_journey', 'month_of_journey','Airline_Jet Airways Business', 'Airline_Multiple carriers', 
                                  'Airline_Air India','Destination_Delhi', 'Dep_Min', 'Arrival_Hour',
                                  'Dep_Hour', 'Arrival_Min', 'Destination_New Delhi','Destination_Hyderabad', 'Source_Mumbai',
                                  'Airline_IndiGo', 'Source_Banglore', 'Destination_Cochin','Source_Delhi']
            categorical_columns = ['Total_Stops'
                
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Price"
            numerical_columns =  ['Duration_Total_Hour', 'Airline_Jet Airways', 'day_of_journey', 'month_of_journey','Airline_Jet Airways Business', 'Airline_Multiple carriers', 
                                  'Airline_Air India','Destination_Delhi', 'Dep_Min', 'Arrival_Hour',
                                  'Dep_Hour', 'Arrival_Min', 'Destination_New Delhi','Destination_Hyderabad', 'Source_Mumbai',
                                  'Airline_IndiGo', 'Source_Banglore', 'Destination_Cochin','Source_Delhi']
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)