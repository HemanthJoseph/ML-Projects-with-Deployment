#training different models and checking their accuracies

#imports
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# class for data training inputs and configuration
@dataclass # decorator, we can directly define class variables
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],  # all except the last column
                train_array[:,-1], #last column
                test_array[:,:-1],
                test_array[:,-1]
            )

            #creating a dictionary of the models
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test=X_test, y_test = y_test, models=models)

            #get the best model score form the dictionary
            best_model_score = max(sorted(model_report.values()))

            #best model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Best model found for both train and test data")

            #save the model path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted_best_model = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted_best_model)

            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)