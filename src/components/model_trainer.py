import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.Exception import CustomException
from src.logger import logging
from src.utills import save_object
import numpy as np


@dataclass
class ModelTrainerConfig:
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    artifacts_dir = os.path.join(base_path, "artifacts")
    trained_model_file_path = os.path.join(artifacts_dir, "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
            }

            model_report = {}
            best_model_name = None
            best_model_score = -float("inf")

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_report[model_name] = score

                logging.info(f"{model_name} R2 Score: {score}")

                if score > best_model_score:
                    best_model_score = score
                    best_model_name = model_name
                    best_model = model

            if best_model_name is None:
                raise CustomException("No model performed well", sys)

            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ Return this instead of None
            return f"Best model: {best_model_name} | R2 Score: {best_model_score:.4f}"

        except Exception as e:
            raise CustomException(e, sys)
