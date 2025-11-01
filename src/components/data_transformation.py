import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.Exception import CustomException
from src.logger import logging

from src.utills import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features=["writing_score","reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            numpipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
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

            logging.info(f"Categorical columns:{categorical_features}")
            logging.info(f"Numerical columns:{numerical_features}")
            

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numpipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)

                
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math score"
            numerical_columns=["writing score","reading score"] 

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) 
            target_feature_test_df=test_df[target_column_name] 

            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)             
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)  

            train_arr=np.c_[input_feature_train_df,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_df,np.array(target_feature_test_df)]

            logging .info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
