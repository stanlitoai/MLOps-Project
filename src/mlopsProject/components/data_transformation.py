import os
import sys
import pandas as pd
from src.exception import CustomException
from mlopsProject import logger 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from mlopsProject.config.configuration import DataIngestion

STAGE_NAME = "Data Transformation  stage 2"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = DataIngestion()
        data_transform_config = config.get_data_transformation()
        



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataTransformationPipeline()
        train_data, test_data = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<<")
        logger.error(f">>>>>> {e} <<<<<<<")