import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

# class for data ingestion inputs and configuration
@dataclass # decorator, we can directly define class variables
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "tain.csv")  # data ingestion will save the data in this location with that file name
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # this will call the above config class and save all path variables as objects
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')  # you cna also read from any other locations, databases etc.
            logging.info("Read the dataset as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_Set, test_Set = train_test_split(df, test_size=0.2, random_state=42)

            train_Set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_Set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException (e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))