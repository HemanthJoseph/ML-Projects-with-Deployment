# For tranformation of categorical and numercial features
# data cleaning
# feature engineering

# imports
import os
import sys
import numpy as np
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # column tranform for data engineering to create a pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# class for data transformation inputs and configuration
@dataclass # decorator, we can directly define class variables
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):  # to create the pickle files to perform data transformation
        """
        This functions is responsible for data transformation
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_cloumns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # create a pipeline and handle missing values
            numerical_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]
            )

            logging.info("Numerical columns standad scaling completed")

            categorical_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder", OneHotEncoder()),
                       ("scaler", StandardScaler(with_mean=False))]

            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipelines", categorical_pipeline, categorical_cloumns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns= [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns= [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(  # used to save the pickle file
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e, sys) # type: ignore
