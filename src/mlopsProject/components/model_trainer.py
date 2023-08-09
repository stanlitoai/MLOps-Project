import pandas as pd
from mlopsProject import logger
from mlopsProject.config.configuration import DataIngestion


STAGE_NAME = "Data Transformation Stage 2"

class ModelTrainingPipeline:
    def main(self):
        try:
            config = DataIngestion()
            
            train_data_path, test_data_path = config.initiate_data_ingestion()
            
            train_arr, test_arr = config.initiate_data_transformation(train_data_path, test_data_path)
            
            print(config.initiate_model_trainer(train_arr,test_arr))
            
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
            
        except Exception as e:
            logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<<")
            logger.error(f">>>>>> {e} <<<<<<<")

if __name__ == "__main__":
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
    
    data_transformation_pipeline = ModelTrainingPipeline()
    data_transformation_pipeline.main()
