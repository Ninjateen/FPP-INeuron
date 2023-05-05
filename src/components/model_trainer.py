import os
import sys
sys.path.append('src')
sys.path.append("src\components")
from exception import CustomException
from logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
import lightgbm as ltb

from sklearn.model_selection import KFold


from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Data_storage","model.pkl")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("Data_storage","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, processor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
                )

            num_folds = 10
            seed = 18

            models = {
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "SVR": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "LGBMRegressor": ltb.LGBMRegressor(),
                "Bagging Regressor": BaggingRegressor()
            }

            params = {
                "Linear Regression": {},
                "KNeighbors Regressor": {},
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "SVR": {},
                "Decision Tree": {},
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LGBMRegressor": {},
                "Bagging Regressor": {}
            }
            
            
            model_report = evaluate_models(X_train=X_train, y_train=y_train,
                                            X_test=X_test, y_test=y_test,
                                            models=models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square,best_model

        except Exception as e:
            raise CustomException(e,sys)
        
        