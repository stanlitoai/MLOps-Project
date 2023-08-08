"""main """
from src.exception import CustomException
from mlopsProject import logger
import sys
from mlopsProject.components.data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage one training"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<<")
    logger.error(f">>>>>> {e} <<<<<<<")



STAGE_NAME = "Data Transformation stage two"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataTransformationPipeline()
    train_data, test_data = obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<<")
    logger.error(f">>>>>> {e} <<<<<<<")