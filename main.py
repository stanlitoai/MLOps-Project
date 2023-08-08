"""main """
from src.exception import CustomException
from mlopsProject import logger
import sys
from mlopsProject.components.data_ingestion import DataIngestionTrainingPipeline
from mlopsProject.components.data_transformation import DataTransformationPipeline
from mlopsProject.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion stage one training"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    train_data, test_data = obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<<")
    logger.error(f">>>>>> {e} <<<<<<<")



STAGE_NAME = "Data Transformation stage two"


try:
    config = DataIngestion()
    
    train_data_path, test_data_path = config.initiate_data_ingestion()
    
    train_arr, test_arr = config.initiate_data_transformation(train_data_path, test_data_path)
    
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
    
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<<")
    logger.error(f">>>>>> {e} <<<<<<<")