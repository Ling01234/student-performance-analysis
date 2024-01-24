import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_train import *


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Reading dataset as dataframe')

            os.makedirs(os.path.dirname(
                self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logging.info('Train test split inititated')
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.config.test_data_path,
                            index=False, header=True)

            logging.info('train and testing split completed')

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data)

    model_trainer = ModelTrainer()
    r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(r2)
