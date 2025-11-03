import os
import sys
from src.Exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass   

from src.components.data_transformation import DataTransformationConfig, dataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer   

# ✅ Always refer to root-level artifacts folder
@dataclass
class DataIngestionConfig:
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Go two levels up to project root
    artifacts_dir = os.path.join(base_path, "artifacts")

    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(
                r"C:\Users\admin\Desktop\MACHINE_LEARNING_CODES\ML_PROJECT\notebook\data\StudentsPerformance.csv"
            )
            logging.info("Read the dataset as DataFrame")

            # ✅ Create root-level artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()  

    data_transformation = dataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data) 

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
