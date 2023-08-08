from pathlib import Path
import os
import sys
import pandas as pd
from mlopsProject.utils.common import read_yaml, create_directories
from mlopsProject.entity.config_entity import DataIngestionConfig
from mlopsProject import logger as Logger
from sklearn.model_selection import train_test_split
from src.exception import CustomException


class DataIngestion:
    def __init__(
        self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        Logger.info(f"Initiating data ingestion method or component")
        try:
            df = pd.read_csv("research/data/student.csv")
            df.head()
            Logger.info(f"Successfully read data from csv file")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            Logger.info(f"Successfully created directories for train and test data")
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            Logger.info(f"Successfully wrote data to csv file")
            Logger.info(f"Train Test Split Initiated")

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path ,index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            Logger.info(f"Successfully wrote data to csv file for train test split")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
                    )
        

        except Exception as e:
            Logger.error(f">>>>>> stage failed <<<<<<<")
            Logger.error(f">>>>>> {e} <<<<<<<")
        