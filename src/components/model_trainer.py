import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.Exception import CustomException
from src.logger import logging

# Assuming save_object and evaluate_model are correctly implemented in src.utills
from src.utills import save_object, evaluate_model 

@dataclass
class ModelTrainerConfig:
    # Adjusted path handling to be more robust
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) 
    artifacts_dir = os.path.join(base_path, "artifacts")
    
    trained_model_file_path: str = os.path.join(artifacts_dir, "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            # Assuming the last column is the target variable
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor( eval_metric='rmse'), # Add eval_metric for warning suppression
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # 🛑 CRITICAL FIX: Updated 'criterion' for Decision Tree Regressor
            # Valid criteria for Regression: 'squared_error', 'friedman_mse', 'absolute_error', 'poisson'
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [10, 20]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2], 
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5]
                },
                "Linear Regression": {},
                "XGB Regressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "CatBoost Regressor": {
                    'iterations': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test,
                                                 models=models,
                                                 params=params)
            
            # Get the best model score
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found (R2 score below 0.6)")
            
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Save the best model object
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) 

            # Calculate R2 score on test data using the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return f"Best Model: {best_model_name}, R2 Score: {r2_square:.4f}" 

        except Exception as e:
            raise CustomException(e, sys)