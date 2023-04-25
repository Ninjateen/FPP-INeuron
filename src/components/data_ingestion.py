import os
import sys
sys.path.append('src')
sys.path.append("src\components")
from exception import CustomException
from logger import logging
import pandas as pd

from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

from model_trainer import ModelTrainerConfig
from model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("Data storage",'train.csv')
    test_data_path: str=os.path.join("Data storage",'test.csv')
    raw_data_path: str=os.path.join("Data storage",'data.csv')
    

class DatIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            
            df = pd.read_csv("DATASET\\cleaned_dataset\\cleaned_dataset.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initiated")
            train_set,test_set= train_test_split(df,test_size=0.2,random_state=35)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
            
            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e: 
               raise CustomException(e,sys) 


if __name__=="__main__":
    obj=DatIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr, "Data storage/preprocessing.pkl"))

