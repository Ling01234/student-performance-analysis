import sys
import os
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.utils import *
from src.logger import logging


# creating config class
@dataclass
class ModelTrainerConfig:
    trainer_model_filepath = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and testing input data')

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Classifier': KNeighborsRegressor(),
                'XGB Classifier': XGBRegressor(),
                'CatBoosting Classifier': CatBoostRegressor(verbose=False),
                'AdaBoost Classifier': AdaBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]

            best_model = models[best_model]

            if best_model_score < 0.6:  # check for performance
                raise CustomException('No best model found')

            logging.info(
                'Best model found on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trainer_model_filepath,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
